# frozen_string_literal: true

# AI Auto-Responder for Mubtkir - مُبتكِر
#
# This service intercepts incoming customer messages, searches the local pgvector
# knowledge base for relevant Daftra documentation context, sends the query + context
# to OpenAI (gpt-4o-mini), and posts the AI response back into the conversation.
#
# If no relevant context is found or the AI cannot answer, it triggers a human
# handoff: sends an Arabic apology message, opens the conversation, and assigns
# it to the default human team.
#
# Usage:
#   AiAutoResponder::RespondService.new(message: message).perform

class AiAutoResponder::RespondService
  MODEL = 'gpt-4o-mini'
  MAX_CONVERSATION_MESSAGES = 20
  MAX_CONTEXT_CHUNKS = 5
  SIMILARITY_THRESHOLD = 0.85         # Max cosine distance (lower = stricter)
  MIN_RELEVANT_CHUNKS = 1             # Minimum chunks needed to attempt AI answer
  HANDOFF_MESSAGE = 'عذراً، سأقوم بتحويلك إلى أحد ممثلي خدمة العملاء لمساعدتك بشكل أفضل.'

  SYSTEM_PROMPT = <<~PROMPT.freeze
    أنت مساعد خدمة عملاء ذكي لشركة "مُبتكِر - Mubtkir".
    مهمتك هي مساعدة العملاء بالإجابة على أسئلتهم حول نظام دفترة (Daftra) للمحاسبة وإدارة الأعمال.

    ## القواعد الأساسية:
    1. **أجب دائماً باللغة العربية** بغض النظر عن لغة السؤال.
    2. كن مهذباً ومحترفاً ومختصراً في إجاباتك.
    3. استخدم المعلومات المقدمة في "سياق قاعدة المعرفة" أدناه للإجابة.
    4. إذا كان السؤال خارج نطاق المعلومات المتاحة، أجب بـ: "HANDOFF_NEEDED"
    5. لا تختلق معلومات غير موجودة في السياق المقدم.
    6. إذا كان السؤال عاماً (تحية، شكر، وداع)، رد بشكل طبيعي ومهذب.
    7. استخدم التنسيق المناسب (قوائم مرقمة، نقاط) لتسهيل القراءة.

    ## معلومات الشركة:
    - اسم الشركة: مُبتكِر (Mubtkir)
    - التخصص: حلول نظام دفترة (Daftra) للمحاسبة وإدارة الأعمال
    - نوع الخدمة: دعم فني واستشارات
  PROMPT

  def initialize(message:)
    @message = message
    @conversation = message.conversation
    @account = @conversation.account
    @inbox = @conversation.inbox
  end

  def perform
    return unless should_respond?

    # Search for relevant context
    context_chunks = search_knowledge_base
    context_text = build_context_text(context_chunks)

    # If no relevant context found and it's not a generic greeting, handoff
    if context_chunks.size < MIN_RELEVANT_CHUNKS && !generic_greeting?
      return perform_handoff
    end

    # Call OpenAI
    ai_response = call_openai(context_text)
    return perform_handoff if ai_response.blank? || ai_response.include?('HANDOFF_NEEDED')

    send_reply(ai_response)
  rescue StandardError => e
    Rails.logger.error("[AiAutoResponder] Error processing message ##{@message.id}: #{e.message}")
    Rails.logger.error(e.backtrace&.first(5)&.join("\n"))
    perform_handoff
  end

  private

  # -----------------------------------------------------------------------
  # Gate checks
  # -----------------------------------------------------------------------

  def should_respond?
    return false unless @message.incoming?
    return false if @message.content.blank?
    return false if @message.private?

    # Only respond on Evolution API / WhatsApp inboxes
    # Evolution API connects via Channel::Api or Channel::Whatsapp
    return false unless evolution_api_inbox?

    # Don't respond if a human agent is already assigned
    return false if @conversation.assignee_id.present?

    # Don't respond to our own AI replies (prevent loops)
    return false if @conversation.messages
                                 .where(message_type: :outgoing)
                                 .where(sender_type: 'AgentBot')
                                 .where('created_at > ?', 30.seconds.ago)
                                 .exists?

    true
  end

  def evolution_api_inbox?
    # Evolution API typically creates inboxes as Channel::Api or Channel::Whatsapp
    @inbox.channel_type.in?(%w[Channel::Api Channel::Whatsapp])
  end

  def generic_greeting?
    greetings = %w[مرحبا مرحباً هلا أهلا السلام سلام hi hello hey مساء صباح شكرا شكراً]
    content = @message.content.strip.downcase
    greetings.any? { |g| content.start_with?(g) || content == g } && content.length < 30
  end

  # -----------------------------------------------------------------------
  # Knowledge Base Search (pgvector)
  # -----------------------------------------------------------------------

  def search_knowledge_base
    query_embedding = generate_embedding(@message.content)
    return [] if query_embedding.blank?

    KnowledgeBaseEmbedding.nearest_neighbors(
      query_embedding,
      limit: MAX_CONTEXT_CHUNKS,
      threshold: SIMILARITY_THRESHOLD
    ).to_a
  rescue StandardError => e
    Rails.logger.error("[AiAutoResponder] KB search error: #{e.message}")
    []
  end

  def generate_embedding(text)
    api_key = fetch_api_key
    return nil if api_key.blank?

    response = HTTParty.post(
      "#{openai_base_url}/v1/embeddings",
      headers: {
        'Content-Type' => 'application/json',
        'Authorization' => "Bearer #{api_key}"
      },
      body: {
        model: 'text-embedding-3-small',
        input: text.truncate(8000),
        dimensions: 1536
      }.to_json,
      timeout: 15
    )

    parsed = JSON.parse(response.body)
    parsed.dig('data', 0, 'embedding')
  rescue StandardError => e
    Rails.logger.error("[AiAutoResponder] Embedding error: #{e.message}")
    nil
  end

  def build_context_text(chunks)
    return '' if chunks.empty?

    sections = chunks.map.with_index do |chunk, i|
      "### مرجع #{i + 1} (#{chunk.source_title})\n#{chunk.content}"
    end

    "## سياق قاعدة المعرفة:\n\n#{sections.join("\n\n")}"
  end

  # -----------------------------------------------------------------------
  # OpenAI Chat Completion
  # -----------------------------------------------------------------------

  def call_openai(context_text)
    messages = build_chat_messages(context_text)
    api_key = fetch_api_key
    return nil if api_key.blank?

    response = HTTParty.post(
      "#{openai_base_url}/v1/chat/completions",
      headers: {
        'Content-Type' => 'application/json',
        'Authorization' => "Bearer #{api_key}"
      },
      body: {
        model: MODEL,
        messages: messages,
        max_tokens: 1024,
        temperature: 0.4
      }.to_json,
      timeout: 30
    )

    parsed = JSON.parse(response.body)
    parsed.dig('choices', 0, 'message', 'content')
  rescue StandardError => e
    Rails.logger.error("[AiAutoResponder] OpenAI error: #{e.message}")
    nil
  end

  def build_chat_messages(context_text)
    messages = []

    # System prompt with knowledge context
    system_content = SYSTEM_PROMPT.dup
    system_content += "\n\n#{context_text}" if context_text.present?
    messages << { role: 'system', content: system_content }

    # Recent conversation history for context continuity
    @conversation.messages
                 .where(message_type: [:incoming, :outgoing])
                 .where(private: false)
                 .order(created_at: :asc)
                 .last(MAX_CONVERSATION_MESSAGES)
                 .each do |msg|
      next if msg.content.blank?

      role = msg.incoming? ? 'user' : 'assistant'
      messages << { role: role, content: msg.content }
    end

    messages
  end

  # -----------------------------------------------------------------------
  # Reply & Handoff
  # -----------------------------------------------------------------------

  def send_reply(content)
    @conversation.messages.create!(
      message_type: :outgoing,
      content: content,
      account_id: @account.id,
      inbox_id: @inbox.id,
      sender: ai_agent_bot
    )
  end

  def perform_handoff
    # 1. Send the handoff message to the customer
    send_reply(HANDOFF_MESSAGE)

    # 2. Open the conversation so human agents can see it
    @conversation.update!(status: :open)

    # 3. Remove the AI bot assignment
    @conversation.update!(assignee_agent_bot_id: nil)

    # 4. Assign to the technical support team or fallback to the first team
    technical_team = @account.teams.find_by(name: 'الدعم الفني')
    default_team = technical_team || @account.teams.first
    
    if default_team
      @conversation.update!(team_id: default_team.id)
      Rails.logger.info("[AiAutoResponder] Handed off conversation ##{@conversation.display_id} to team '#{default_team.name}'")
    else
      Rails.logger.warn("[AiAutoResponder] No team found for handoff in account ##{@account.id}")
    end
  end

  # -----------------------------------------------------------------------
  # Helpers
  # -----------------------------------------------------------------------

  def ai_agent_bot
    @ai_agent_bot ||= AgentBot.find_or_create_by!(name: 'Mubtkir AI') do |bot|
      bot.description = 'مساعد مُبتكِر الذكي - AI Auto-responder'
    end
  end

  def fetch_api_key
    # Priority: 1) Account-level OpenAI hook, 2) System config, 3) ENV
    hook_key = @account.hooks.find_by(app_id: 'openai', status: 'enabled')&.settings&.dig('api_key')
    return hook_key if hook_key.present?

    system_key = InstallationConfig.find_by(name: 'CAPTAIN_OPEN_AI_API_KEY')&.value
    return system_key if system_key.present?

    ENV.fetch('OPENAI_API_KEY', nil)
  end

  def openai_base_url
    base = InstallationConfig.find_by(name: 'CAPTAIN_OPEN_AI_ENDPOINT')&.value.presence || 'https://api.openai.com'
    base.chomp('/')
  end
end
