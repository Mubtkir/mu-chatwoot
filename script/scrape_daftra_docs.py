#!/usr/bin/env python3
"""
Daftra Documentation Scraper & Embedder
========================================
Scrapes the Daftra (دفترة) help center documentation, chunks the text,
generates embeddings via OpenAI, and upserts them into the pgvector
knowledge_base_embeddings table.

Designed to run as a cron job every 6 months:
  0 0 1 */6 * /path/to/venv/bin/python /path/to/scrape_daftra_docs.py

Requirements:
  pip install requests beautifulsoup4 psycopg2-binary openai tiktoken

Environment Variables:
  OPENAI_API_KEY       - Your OpenAI API key
  DATABASE_URL         - PostgreSQL connection string (e.g. postgresql://user:pass@host:5432/chatwoot_production)
  DAFTRA_BASE_URL      - (optional) Override base URL, default: https://daftra.com/ar/help
"""

import os
import sys
import hashlib
import time
import logging
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
import tiktoken

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
DAFTRA_BASE_URL = os.environ.get("DAFTRA_BASE_URL", "https://daftra.com/ar/help")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS_PER_CHUNK = 500          # ~2000 chars, fits well in context
CHUNK_OVERLAP_TOKENS = 50           # Overlap for context continuity
REQUEST_DELAY = 1.0                 # Seconds between HTTP requests (polite scraping)
BATCH_SIZE = 100                    # Embeddings per API call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI Embedding Client
# ---------------------------------------------------------------------------
class EmbeddingClient:
    """Minimal OpenAI embeddings client using the REST API directly."""

    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        response = self.session.post(
            self.API_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": texts,
                "dimensions": EMBEDDING_DIMENSIONS,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split text into overlapping chunks based on token count."""
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        start += max_tokens - overlap

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Web Scraper
# ---------------------------------------------------------------------------
class DaftraScraper:
    """Crawls the Daftra help center and extracts article content."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MubtikirBot/1.0 (Knowledge Base Scraper)",
            "Accept-Language": "ar,en;q=0.5",
        })
        self.visited = set()
        self.articles = []

    def scrape(self) -> list[dict]:
        """Main entry point: discover and scrape all help articles."""
        logger.info(f"Starting scrape from: {self.base_url}")
        self._discover_links(self.base_url)
        logger.info(f"Discovered {len(self.visited)} pages to scrape")

        for url in sorted(self.visited):
            try:
                article = self._scrape_page(url)
                if article:
                    self.articles.append(article)
                    logger.info(f"  ✓ Scraped: {article['title'][:60]}...")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logger.warning(f"  ✗ Failed to scrape {url}: {e}")

        logger.info(f"Successfully scraped {len(self.articles)} articles")
        return self.articles

    def _discover_links(self, url: str, depth: int = 0, max_depth: int = 3):
        """Recursively discover help article links."""
        if depth > max_depth or url in self.visited:
            return

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning(f"  Failed to fetch {url}: {e}")
            return

        soup = BeautifulSoup(resp.text, "html.parser")
        base_domain = urlparse(self.base_url).netloc

        for link in soup.find_all("a", href=True):
            href = urljoin(url, link["href"])
            parsed = urlparse(href)

            # Only follow links within the help section of the same domain
            if (parsed.netloc == base_domain and
                "/help" in parsed.path and
                href not in self.visited and
                "#" not in href):

                clean_url = href.split("?")[0].split("#")[0]
                self.visited.add(clean_url)
                self._discover_links(clean_url, depth + 1, max_depth)

    def _scrape_page(self, url: str) -> dict | None:
        """Extract title and body text from a single help page."""
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try common article content selectors
        title_el = soup.find("h1") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else "Untitled"

        # Try to find the main content area
        content_el = (
            soup.find("article") or
            soup.find("div", class_=re.compile(r"(content|article|post|entry|body)", re.I)) or
            soup.find("main") or
            soup.find("div", {"role": "main"})
        )

        if not content_el:
            return None

        # Remove navigation, sidebar, footer elements
        for tag in content_el.find_all(["nav", "footer", "aside", "script", "style", "header"]):
            tag.decompose()

        text = content_el.get_text(separator="\n", strip=True)

        # Skip very short pages (likely index/nav pages)
        if len(text) < 100:
            return None

        return {
            "title": title,
            "url": url,
            "text": text,
        }


# ---------------------------------------------------------------------------
# Database Operations
# ---------------------------------------------------------------------------
class EmbeddingStore:
    """Manages the pgvector knowledge_base_embeddings table."""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = False

    def upsert_chunks(self, chunks: list[dict]):
        """
        Upsert chunks with their embeddings into the database.
        Uses chunk_hash for deduplication (ON CONFLICT DO UPDATE).

        Each chunk dict should have:
          - content: str
          - source_url: str
          - source_title: str
          - chunk_index: int
          - chunk_hash: str
          - embedding: list[float]
        """
        if not chunks:
            return

        sql = """
            INSERT INTO knowledge_base_embeddings
                (content, source_url, source_title, chunk_index, chunk_hash, embedding, created_at, updated_at)
            VALUES %s
            ON CONFLICT (chunk_hash)
            DO UPDATE SET
                content = EXCLUDED.content,
                source_url = EXCLUDED.source_url,
                source_title = EXCLUDED.source_title,
                embedding = EXCLUDED.embedding,
                updated_at = EXCLUDED.updated_at
        """
        now = "NOW()"
        values = []
        for chunk in chunks:
            embedding_str = f"[{','.join(str(v) for v in chunk['embedding'])}]"
            values.append((
                chunk["content"],
                chunk["source_url"],
                chunk["source_title"],
                chunk["chunk_index"],
                chunk["chunk_hash"],
                embedding_str,
            ))

        # psycopg2 execute_values with custom template for NOW()
        template = "(%s, %s, %s, %s, %s, %s::vector, NOW(), NOW())"
        with self.conn.cursor() as cur:
            execute_values(cur, sql, values, template=template, page_size=100)
        self.conn.commit()
        logger.info(f"  Upserted {len(chunks)} chunks into database")

    def get_existing_hashes(self) -> set[str]:
        """Return all existing chunk_hashes for change detection."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT chunk_hash FROM knowledge_base_embeddings")
            return {row[0] for row in cur.fetchall()}

    def delete_stale(self, active_hashes: set[str]):
        """Remove chunks whose hashes are no longer in the active set."""
        existing = self.get_existing_hashes()
        stale = existing - active_hashes
        if stale:
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM knowledge_base_embeddings WHERE chunk_hash = ANY(%s)",
                    (list(stale),)
                )
            self.conn.commit()
            logger.info(f"  Deleted {len(stale)} stale chunks")

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Daftra Documentation Scraper & Embedder")
    logger.info("=" * 60)

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    # 1. Scrape documentation
    scraper = DaftraScraper(DAFTRA_BASE_URL)
    articles = scraper.scrape()

    if not articles:
        logger.warning("No articles found. Exiting.")
        return

    # 2. Chunk all articles
    logger.info("Chunking articles...")
    all_chunks = []
    for article in articles:
        text_chunks = chunk_text(article["text"])
        for i, chunk in enumerate(text_chunks):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            all_chunks.append({
                "content": chunk,
                "source_url": article["url"],
                "source_title": article["title"],
                "chunk_index": i,
                "chunk_hash": chunk_hash,
            })
    logger.info(f"Created {len(all_chunks)} chunks from {len(articles)} articles")

    # 3. Generate embeddings in batches
    logger.info("Generating embeddings...")
    client = EmbeddingClient(OPENAI_API_KEY)
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        try:
            embeddings = client.embed(texts)
            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
            logger.info(
                f"  Embedded batch {batch_start // BATCH_SIZE + 1}"
                f"/{(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}"
            )
        except Exception as e:
            logger.error(f"  Embedding API error: {e}")
            sys.exit(1)

        time.sleep(0.5)  # Rate limit courtesy

    # 4. Upsert into database
    logger.info("Upserting into pgvector database...")
    store = EmbeddingStore(DATABASE_URL)
    try:
        # Upsert in batches
        for batch_start in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[batch_start:batch_start + BATCH_SIZE]
            store.upsert_chunks(batch)

        # Remove stale chunks (from pages that no longer exist)
        active_hashes = {c["chunk_hash"] for c in all_chunks}
        store.delete_stale(active_hashes)
    finally:
        store.close()

    logger.info("=" * 60)
    logger.info(f"✓ Pipeline complete: {len(all_chunks)} chunks indexed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
