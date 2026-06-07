# frozen_string_literal: true

# == Schema Information
#
# Table name: knowledge_base_embeddings
#
#  id           :bigint           not null, primary key
#  content      :text             not null
#  source_url   :text
#  source_title :string
#  chunk_hash   :string           not null
#  chunk_index  :integer          default(0)
#  embedding    :vector(1536)
#  created_at   :datetime         not null
#  updated_at   :datetime         not null
#
# Indexes
#
#  index_knowledge_base_embeddings_on_chunk_hash  (chunk_hash) UNIQUE
#  index_knowledge_base_embeddings_on_source_url  (source_url)
#  idx_embeddings_vector                          (embedding) USING ivfflat
#

class KnowledgeBaseEmbedding < ApplicationRecord
  validates :content, presence: true
  validates :chunk_hash, presence: true, uniqueness: true

  # Perform cosine similarity search against the embeddings table.
  # Returns the top `limit` most relevant chunks for the given query embedding.
  #
  # @param query_embedding [Array<Float>] The embedding vector of the user's query
  # @param limit [Integer] Maximum number of results to return
  # @param threshold [Float] Maximum cosine distance (lower = more similar, 0..2 range)
  # @return [ActiveRecord::Relation] Ordered by similarity (most similar first)
  scope :nearest_neighbors, lambda { |query_embedding, limit: 5, threshold: 1.0|
    embedding_literal = "[#{query_embedding.join(',')}]"
    where("embedding <=> '#{embedding_literal}' < ?", threshold)
      .order(Arel.sql("embedding <=> '#{embedding_literal}'"))
      .limit(limit)
      .select("*, (embedding <=> '#{embedding_literal}') AS distance")
  }
end
