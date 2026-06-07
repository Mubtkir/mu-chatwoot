# frozen_string_literal: true

class AiAutoResponderListener < BaseListener
  def message_created(event)
    message = extract_message_and_account(event)[0]

    # Only process incoming customer messages
    return unless message.incoming?
    return if message.content.blank?
    return if message.private?

    # Enqueue the AI responder job asynchronously
    AiAutoResponder::RespondJob.perform_later(message.id)
  end
end
