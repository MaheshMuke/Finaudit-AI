import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

COMMODITY_REGISTRY = {
    "Brent Crude": {"symbol": "BZ=F", "unit": "$/barrel", "category": "Energy"},
    "WTI Crude": {"symbol": "CL=F", "unit": "$/barrel", "category": "Energy"},
    "Natural Gas": {"symbol": "NG=F", "unit": "$/MMBtu", "category": "Energy"},
    "Gold": {"symbol": "GC=F", "unit": "$/troy oz", "category": "Metals"},
    "Silver": {"symbol": "SI=F", "unit": "$/troy oz", "category": "Metals"},
    "Copper": {"symbol": "HG=F", "unit": "$/lb", "category": "Industrial Metals"},
    "USD / INR": {"symbol": "USDINR=X", "unit": "INR", "category": "Currencies"},
}

SECTOR_COMMODITY_EXPOSURE = {
    "Oil Gas & Consumable Fuels": ["Brent Crude", "WTI Crude", "Natural Gas", "USD / INR"],
    "Chemicals": ["Brent Crude", "Natural Gas", "USD / INR"],
    "Automobile and Auto Components": ["Copper", "USD / INR", "Brent Crude"],
    "Metals & Mining": ["Copper", "Gold", "Silver", "USD / INR"],
    "Power": ["Natural Gas", "Brent Crude"],
    "Information Technology": ["USD / INR"],  # Significant USD revenue sensitivity
    "Consumer Durables": ["Gold", "Silver", "Copper"],
    "Financial Services": ["USD / INR"],
}


class CommodityService:
    """
    Market indicators for major global commodity futures and currency pairs via yfinance.
    Provides current prices, 5-day changes, and sector-to-commodity mapping.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_indicators(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch latest price and 5-day percentage change for tracked commodities."""
        results = []
        for name, meta in COMMODITY_REGISTRY.items():
            sym = meta["symbol"]
            if not force_refresh and sym in self._cache:
                results.append(self._cache[sym])
                continue

            item = {
                "name": name,
                "symbol": sym,
                "unit": meta["unit"],
                "category": meta["category"],
                "last_price": None,
                "change_pct": None,
                "status": "online",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }

            if yf is None:
                item["status"] = "unavailable"
                self._cache[sym] = item
                results.append(item)
                continue

            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d")
                if not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].dropna()
                    if len(closes) > 0:
                        curr = float(closes.iloc[-1])
                        prev = float(closes.iloc[0])
                        item["last_price"] = round(curr, 2)
                        if prev > 0:
                            item["change_pct"] = round(((curr - prev) / prev) * 100, 2)
            except Exception as e:
                logger.warning("Failed fetching commodity %s: %s", sym, e)
                item["status"] = "unavailable"

            self._cache[sym] = item
            results.append(item)

        return results

    def get_sector_commodities(self, industry_or_sector: str) -> List[Dict[str, Any]]:
        """Find relevant commodity indicators for a given company industry."""
        matched_keys = SECTOR_COMMODITY_EXPOSURE.get(industry_or_sector, [])
        if not matched_keys:
            # Fuzzy match
            for sec_key, com_list in SECTOR_COMMODITY_EXPOSURE.items():
                if sec_key.lower() in industry_or_sector.lower() or industry_or_sector.lower() in sec_key.lower():
                    matched_keys = com_list
                    break

        all_indicators = self.get_indicators()
        return [ind for ind in all_indicators if ind["name"] in matched_keys]


_commodity_instance: Optional[CommodityService] = None


def get_commodity_service() -> CommodityService:
    global _commodity_instance
    if _commodity_instance is None:
        _commodity_instance = CommodityService()
    return _commodity_instance
