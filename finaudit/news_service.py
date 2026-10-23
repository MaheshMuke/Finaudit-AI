import hashlib
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import feedparser
except ImportError:
    feedparser = None
from finaudit.company_registry import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "category": "Markets",
    },
    {
        "name": "Economic Times Industry",
        "url": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
        "category": "Industry",
    },
    {
        "name": "Livemint Markets",
        "url": "https://www.livemint.com/rss/markets",
        "category": "Markets",
    },
    {
        "name": "Business Standard Top",
        "url": "https://www.business-standard.com/rss/home_page_top_stories.rss",
        "category": "Economy",
    },
]


class NewsService:
    """
    Provider-independent RSS news collection, deduplication, SQLite caching,
    and keyword/relevance filtering for Indian financial markets.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (ROOT_DIR / "data" / "finaudit.sqlite3")
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rss_articles (
                    article_id TEXT PRIMARY KEY,
                    source_name TEXT,
                    category TEXT,
                    title TEXT NOT NULL,
                    summary TEXT,
                    link TEXT UNIQUE,
                    published_at TEXT,
                    created_at TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rss_created ON rss_articles(created_at);")

    def fetch_latest_feeds(self, max_items_per_feed: int = 15) -> int:
        """Fetch configured RSS feeds and persist new articles in SQLite."""
        new_count = 0
        now = datetime.now(timezone.utc).isoformat()

        if feedparser is None:
            logger.info("feedparser not installed, skipping live RSS feed fetching")
            return 0

        for feed_cfg in DEFAULT_FEEDS:
            try:
                parsed = feedparser.parse(
                    feed_cfg["url"],
                    request_headers={"User-Agent": "FinAuditAI/0.2 (Research Aggregator; Respectful Rate Limits)"},
                )
                for entry in parsed.entries[:max_items_per_feed]:
                    link = getattr(entry, "link", "").strip()
                    title = getattr(entry, "title", "").strip()
                    if not link or not title:
                        continue

                    # Stable deterministic hash based on link/title
                    art_id = hashlib.sha256(link.encode("utf-8")).hexdigest()[:16]
                    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                    pub = getattr(entry, "published", "") or now

                    with sqlite3.connect(self.db_path) as conn:
                        try:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO rss_articles
                                (article_id, source_name, category, title, summary, link, published_at, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (art_id, feed_cfg["name"], feed_cfg["category"], title, summary, link, pub, now),
                            )
                            if conn.total_changes > 0:
                                new_count += 1
                        except Exception:
                            pass
            except Exception as e:
                logger.warning("Feed fetch failed for %s: %s", feed_cfg["name"], e)

        return new_count

    def search_news(
        self,
        keywords: List[str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query articles containing company names, symbols, or sector keywords."""
        if not keywords:
            return self.get_recent_articles(limit=limit)

        # Ensure we have fresh articles
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            count = conn.execute("SELECT COUNT(*) as c FROM rss_articles").fetchone()["c"]

        if count < 5:
            self.fetch_latest_feeds()

        pattern_clauses = []
        params = []
        for kw in keywords:
            clean = kw.strip()
            if len(clean) >= 3:
                pattern_clauses.append("(title LIKE ? OR summary LIKE ?)")
                params.extend([f"%{clean}%", f"%{clean}%"])

        if not pattern_clauses:
            return self.get_recent_articles(limit=limit)

        query = f"""
            SELECT article_id, source_name, category, title, summary, link, published_at
            FROM rss_articles
            WHERE {' OR '.join(pattern_clauses)}
            ORDER BY created_at DESC LIMIT ?
        """
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def get_recent_articles(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT article_id, source_name, category, title, summary, link, published_at
                FROM rss_articles ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


_news_instance: Optional[NewsService] = None


def get_news_service() -> NewsService:
    global _news_instance
    if _news_instance is None:
        _news_instance = NewsService()
    return _news_instance
