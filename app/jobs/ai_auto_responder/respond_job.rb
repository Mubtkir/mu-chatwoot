# frozen_string_literal: true

class AiAutoResponder::RespondJob < ApplicationJob
  queue_as :default

  # Retry up to 2 times with exponential backoff if OpenAI is temporarily unavailable
  retry_on StandardError, wait: :polynomially_longer, attempts: 2

  def perform(message_id)
    message = Message.find_by(id: message_id)
    return if message.blank?

    AiAutoResponder::RespondService.new(message: message).perform
  end
end
