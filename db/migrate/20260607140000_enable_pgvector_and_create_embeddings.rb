# frozen_string_literal: true

class EnablePgvectorAndCreateEmbeddings < ActiveRecord::Migration[7.0]
  def up
    # Enable the pgvector extension (requires pgvector installed on the PostgreSQL server)
    # Install on Ubuntu/Debian: sudo apt install postgresql-16-pgvector
    # Install on macOS:         brew install pgvector
    enable_extension 'vector'

    create_table :knowledge_base_embeddings do |t|
      t.text :content, null: false              # The original text chunk
      t.text :source_url                        # URL the chunk was scraped from
      t.string :source_title                    # Page title for reference
      t.string :chunk_hash, null: false         # SHA256 of content for dedup
      t.integer :chunk_index, default: 0        # Order within the source page
      t.column :embedding, 'vector(1536)'       # OpenAI text-embedding-3-small dimension

      t.timestamps
    end

    add_index :knowledge_base_embeddings, :chunk_hash, unique: true
    add_index :knowledge_base_embeddings, :source_url

    # Create an HNSW index for fast approximate nearest neighbor search
    # HNSW builds dynamically from scratch perfectly, avoiding IVFFlat's need for training data.
    execute <<-SQL
      CREATE INDEX idx_embeddings_vector ON knowledge_base_embeddings
      USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
    SQL
  end

  def down
    drop_table :knowledge_base_embeddings
    disable_extension 'vector'
  end
end
