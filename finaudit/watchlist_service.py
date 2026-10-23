import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from finaudit.company_registry import ROOT_DIR


class WatchlistService:
    """
    Manages persistent watchlists and optional portfolio holdings in SQLite.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (ROOT_DIR / "data" / "finaudit.sqlite3")
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlists (
                    watchlist_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watchlist_id TEXT,
                    symbol TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    shares REAL DEFAULT 0,
                    avg_cost REAL DEFAULT 0,
                    notes TEXT DEFAULT '',
                    created_at TEXT,
                    UNIQUE(watchlist_id, symbol)
                );
                """
            )
            # Create default watchlist if none exists
            row = conn.execute("SELECT COUNT(*) FROM watchlists").fetchone()[0]
            if row == 0:
                conn.execute(
                    "INSERT INTO watchlists (watchlist_id, name, created_at) VALUES (?, ?, ?)",
                    ("default", "My Primary Watchlist", datetime.now(timezone.utc).isoformat()),
                )

    def get_watchlists(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM watchlists ORDER BY created_at ASC").fetchall()
        return [dict(r) for r in rows]

    def create_watchlist(self, name: str) -> str:
        wl_id = name.lower().replace(" ", "-")[:20]
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO watchlists (watchlist_id, name, created_at) VALUES (?, ?, ?)",
                (wl_id, name, now),
            )
        return wl_id

    def get_items(self, watchlist_id: str = "default", **kwargs) -> List[Dict[str, Any]]:
        target_id = kwargs.get("list_name", watchlist_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM watchlist_items WHERE watchlist_id = ? ORDER BY created_at DESC",
                (target_id,),
            ).fetchall()
        return [
            {
                "symbol": row["symbol"],
                "company_name": row["company_name"],
                "list_name": row["watchlist_id"],
                "quantity": row["shares"],
                "average_buy_price": row["avg_cost"],
                "notes": row["notes"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def add_item(
        self,
        symbol: str,
        company_name: str,
        watchlist_id: str = "default",
        shares: float = 0,
        avg_cost: float = 0,
        notes: str = "",
        **kwargs,
    ):
        target_id = kwargs.get("list_name", watchlist_id)
        qty = kwargs.get("quantity", shares) or 0
        cost = kwargs.get("average_buy_price", avg_cost) or 0
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO watchlist_items
                (watchlist_id, symbol, company_name, shares, avg_cost, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (target_id, symbol, company_name, qty, cost, notes, now),
            )

    def remove_item(self, symbol: str, watchlist_id: str = "default", **kwargs):
        target_id = kwargs.get("list_name", watchlist_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM watchlist_items WHERE watchlist_id = ? AND symbol = ?",
                (target_id, symbol),
            )


_watchlist_instance: Optional[WatchlistService] = None


def get_watchlist_service() -> WatchlistService:
    global _watchlist_instance
    if _watchlist_instance is None:
        _watchlist_instance = WatchlistService()
    return _watchlist_instance
