import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
NIFTY500_CSV = ROOT_DIR / "ind_nifty500list.csv"
OUTPUT_DIR = ROOT_DIR / "output"
CACHE_DIR = ROOT_DIR / "cache"


class CompanyRecord(BaseModel):
    symbol: str
    company_name: str
    industry: str
    series: str = "EQ"
    isin: str = ""
    yahoo_ticker: str
    business_model: str = ""
    segments: str = ""
    geography: str = ""
    therapeutic_focus: str = ""


# Custom ticker overrides for specific companies (e.g. Tata Motors current restructuring / symbols)
TICKER_OVERRIDES = {
    "TMCV": "TMCV.NS",
    "TMPV": "TMPV.NS",
    "TATAMOTORS": "TMPV.NS",  # Resolve Tata Motors to active passenger/commercial vehicle ticker
    "M&M": "M&M.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "L&T": "LT.NS",
}

# Alias mapping for popular user queries
COMMON_ALIASES = {
    "tata consultancy services": "TCS",
    "tata consultancy": "TCS",
    "tcs": "TCS",
    "reliance": "RELIANCE",
    "reliance industries": "RELIANCE",
    "ril": "RELIANCE",
    "infosys": "INFY",
    "infy": "INFY",
    "hdfc bank": "HDFCBANK",
    "hdfc": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",
    "state bank of india": "SBIN",
    "sbi": "SBIN",
    "tata motors": "TMPV",  # Current NSE symbol for Tata Motors Passenger Vehicles
    "tata motors passenger vehicles": "TMPV",
    "tata motors commercial vehicles": "TMCV",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "wipro": "WIPRO",
    "hcl tech": "HCLTECH",
    "hcl technologies": "HCLTECH",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "itc": "ITC",
    "larsen & toubro": "LT",
    "l&t": "LT",
    "maruti": "MARUTI",
    "maruti suzuki": "MARUTI",
    "kotak bank": "KOTAKBANK",
    "kotak mahindra bank": "KOTAKBANK",
    "axis bank": "AXISBANK",
    "hindustan copper": "HINDCOPPER",
    "hindalco industries": "HINDALCO",
    "hindalco": "HINDALCO",
}

CURATED_PEER_GROUPS = {
    # Reliance spans refining, petrochemicals, energy, and consumer businesses;
    # these listed integrated energy companies are a more defensible company-wide
    # comparison set than small single-operator firms.
    "RELIANCE": ("ONGC", "BPCL", "IOC", "GAIL"),
    # Asset management companies — prefer listed AMCs over banks or insurers
    "NAM-INDIA": ("ABSLAMC", "ICICIAMC", "HDFCAMC", "UTIAMC"),
    "ABSLAMC": ("NAM-INDIA", "ICICIAMC", "HDFCAMC", "UTIAMC"),
    "ICICIAMC": ("NAM-INDIA", "ABSLAMC", "HDFCAMC", "UTIAMC"),
    "HDFCAMC": ("NAM-INDIA", "ABSLAMC", "ICICIAMC", "UTIAMC"),
    "UTIAMC": ("NAM-INDIA", "ABSLAMC", "ICICIAMC", "HDFCAMC"),
}

# -------------------------------------------------------------------------
# Business-model sub-type taxonomy for Financial Services companies.
# Each entry maps a sub-type label to keyword signals extracted from the
# company name and/or business summary. The scoring engine uses this to
# distinguish asset managers, wealth managers, lenders, insurers, and
# other financial businesses before applying the generic profile score.
# -------------------------------------------------------------------------
FINANCIAL_SUBTYPES: Dict[str, List[str]] = {
    "asset_manager": [
        "asset management", "amc", "mutual fund", "fund management",
        "nippon life", "hdfcamc", "utiamc", "abslamc", "iciciamc",
        "mirae asset", "invesco mutual", "pgim india",
    ],
    "wealth_manager": [
        "wealth management", "wealth", "nuvama", "anand rathi wealth",
        "iifl wealth", "360 one", "white oak",
    ],
    "stock_exchange": [
        "stock exchange", "exchange", "bse", "nse", "mcx",
        "clearing corporation", "depository", "cdsl", "nsdl",
    ],
    "insurance": [
        "insurance", "life insurance", "general insurance", "reinsurance",
        "lic", "hdfc life", "sbi life", "icici lombard",
        "star health", "new india assurance", "gic re",
    ],
    "bank": [
        "bank", "banking",
    ],
    "housing_finance": [
        "housing finance", "home loan", "mortgage",
    ],
    "nbfc": [
        "nbfc", "non-banking financial", "finance company",
        "bajaj finance", "muthoot", "manappuram", "shriram",
        "piramal finance", "poonawalla", "capri global",
        "five-star business finance", "creditaccess",
    ],
    "payment": [
        "payment", "fintech", "paytm", "pine labs", "policy bazaar",
        "pb fintech", "one97", "wallet",
    ],
    "government_finance": [
        "power finance", "rec ltd", "irfc", "ireda",
        "hudco", "ifci", "sammaan",
    ],
    "registrar": [
        "computer age management", "cams", "kfin technologies",
        "registrar", "transfer agent",
    ],
    "rating_agency": [
        "crisil", "icra", "care ratings", "credit rating",
    ],
}


class CompanyRegistry:
    def __init__(self, csv_path: Path = NIFTY500_CSV):
        self.csv_path = csv_path
        self._companies: Dict[str, CompanyRecord] = {}
        self._name_index: Dict[str, str] = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            return

        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            sym = str(row.get("Symbol", "")).strip().upper()
            if not sym or sym == "NAN":
                continue
            name = str(row.get("Company Name", "")).strip()
            ind = str(row.get("Industry", "")).strip()
            series = str(row.get("Series", "EQ")).strip()
            isin = str(row.get("ISIN Code", "")).strip()

            # Determine Yahoo ticker
            yahoo_ticker = TICKER_OVERRIDES.get(sym, f"{sym}.NS")

            rec = CompanyRecord(
                symbol=sym,
                company_name=name,
                industry=ind,
                series=series,
                isin=isin,
                yahoo_ticker=yahoo_ticker,
            )
            self._companies[sym] = rec
            norm_name = re.sub(r"[^a-z0-9]", "", name.lower())
            self._name_index[norm_name] = sym

    def get_by_symbol(self, symbol: str) -> Optional[CompanyRecord]:
        sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        if sym in self._companies:
            return self._companies[sym]
        # Check alias
        if sym.lower() in COMMON_ALIASES:
            mapped_sym = COMMON_ALIASES[sym.lower()]
            return self._companies.get(mapped_sym)
        return None

    def resolve(self, query: str) -> Optional[CompanyRecord]:
        q = query.strip()
        if not q:
            return None

        # 1. Exact alias match
        norm_q = q.lower().strip()
        if norm_q in COMMON_ALIASES:
            return self.get_by_symbol(COMMON_ALIASES[norm_q])

        # 2. Check as ticker / symbol
        clean_sym = q.upper().replace(".NS", "").replace(".BO", "").strip()
        if clean_sym in self._companies:
            return self._companies[clean_sym]

        # 3. Normalized full name exact match
        clean_name = re.sub(r"[^a-z0-9]", "", norm_q)
        if clean_name in self._name_index:
            return self._companies[self._name_index[clean_name]]

        # 4. Prefix or substring search
        candidates = self.search(q, limit=1)
        if candidates:
            return candidates[0]

        return None

    def search(self, query: str, limit: int = 10) -> List[CompanyRecord]:
        q = query.strip().lower()
        if not q:
            return []

        # Check aliases first
        if q in COMMON_ALIASES:
            rec = self.get_by_symbol(COMMON_ALIASES[q])
            if rec:
                return [rec]

        clean_q = re.sub(r"[^a-z0-9]", "", q)
        exact_symbol_matches = []
        prefix_matches = []
        substring_matches = []

        for sym, rec in self._companies.items():
            s_lower = sym.lower()
            name_lower = rec.company_name.lower()
            clean_rec_name = re.sub(r"[^a-z0-9]", "", name_lower)

            if s_lower == q or s_lower == q.replace(".ns", ""):
                exact_symbol_matches.append(rec)
            elif s_lower.startswith(q) or name_lower.startswith(q):
                prefix_matches.append(rec)
            elif clean_q in s_lower or clean_q in clean_rec_name:
                substring_matches.append(rec)

        results = exact_symbol_matches + prefix_matches + substring_matches
        seen = set()
        deduped = []
        for r in results:
            if r.symbol not in seen:
                seen.add(r.symbol)
                deduped.append(r)
            if len(deduped) >= limit:
                break
        return deduped

    def resolve_mentions(self, text: str) -> List[CompanyRecord]:
        """Resolve explicit company names and tickers mentioned in a question."""
        normalized = text.lower()
        candidates: List[tuple[int, int, CompanyRecord]] = []
        seen = set()
        aliases = sorted(COMMON_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
        for alias, symbol in aliases:
            match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized)
            if match:
                company = self.get_by_symbol(symbol)
                if company and company.symbol not in seen:
                    candidates.append((match.start(), len(alias), company))
                    seen.add(company.symbol)
        for symbol, company in self._companies.items():
            match = re.search(rf"(?<![a-z0-9]){re.escape(symbol.lower())}(?![a-z0-9])", normalized)
            if match:
                if symbol not in seen:
                    candidates.append((match.start(), len(symbol), company))
                    seen.add(symbol)
        for company in self._companies.values():
            name = company.company_name.lower().strip()
            match = re.search(
                rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", normalized
            ) if len(name) >= 5 else None
            if match and company.symbol not in seen:
                candidates.append((match.start(), len(name), company))
                seen.add(company.symbol)
        return [
            company
            for _, _, company in sorted(candidates, key=lambda item: (item[0], -item[1]))
        ]

    def list_all(self) -> List[CompanyRecord]:
        return list(self._companies.values())

    @staticmethod
    def infer_financial_subtype(company: CompanyRecord, business_summary: str = "") -> str:
        """Infer the financial-services sub-type (e.g. 'asset_manager', 'bank') from
        the company name and optional business summary.

        Returns the most specific matching sub-type key from FINANCIAL_SUBTYPES, or
        'other_financial' when no signal matches.
        """
        text = " ".join([
            company.company_name,
            company.symbol,
            company.business_model,
            company.segments,
            business_summary,
        ]).lower()
        for subtype, keywords in FINANCIAL_SUBTYPES.items():
            if any(kw in text for kw in keywords):
                return subtype
        return "other_financial"

    def get_peers(self, target: CompanyRecord, max_peers: int = 4) -> List[CompanyRecord]:
        """Return baseline same-industry candidates for compatibility."""
        if not target.industry:
            return []

        curated_symbols = CURATED_PEER_GROUPS.get(target.symbol)
        if curated_symbols:
            peers = [
                self._companies[symbol]
                for symbol in curated_symbols
                if symbol in self._companies and symbol != target.symbol
            ]
        else:
            peers = [
                c for c in self._companies.values()
                if c.symbol != target.symbol and c.industry == target.industry
            ]

        # Prefer peers with local financial evidence, then stable industry matches.
        def peer_priority(c: CompanyRecord) -> tuple[int, str]:
            cache_file = CACHE_DIR / f"{c.symbol}.csv"
            return (
                0 if cache_file.exists() and cache_file.stat().st_size > 500 else 1,
                c.company_name,
            )

        if not curated_symbols:
            peers.sort(key=peer_priority)
        return peers[:max_peers]

    def rank_peers(
        self,
        target: CompanyRecord,
        evidence: Optional[Dict[str, Dict[str, object]]] = None,
        max_peers: int = 4,
        candidates: Optional[List[CompanyRecord]] = None,
    ) -> List[Dict[str, object]]:
        """Rank candidates using available business, scale, and data evidence.

        For Financial Services companies the algorithm applies a two-level scoring
        approach:
          1. Infer the financial sub-type (asset_manager, bank, insurer, etc.) for
             both the target and each candidate from the company name and business
             summary.  Matching sub-types receive a strong similarity bonus;
             mismatching sub-types (e.g. bank vs. AMC) receive a penalty that
             prevents clearly unrelated models from being ranked as strong peers.
          2. The legacy cross-sector profile groups (pharma, IT, energy, …) still
             apply for non-financial industries.

        Missing business metadata lowers confidence but never creates a segment claim.
        """
        evidence = evidence or {}
        if candidates is None:
            candidates = self.get_peers(target, max_peers=len(self._companies))
        target_evidence = evidence.get(target.symbol, {})
        target_summary = str(target_evidence.get("business_summary", ""))
        target_subtype = self.infer_financial_subtype(target, target_summary)
        is_financial_industry = "financial" in target.industry.lower()

        def broad_profile(company: CompanyRecord, item: Dict) -> Set[str]:
            raw = " ".join(
                str(value)
                for value in (
                    company.business_model,
                    company.segments,
                    company.geography,
                    company.therapeutic_focus,
                    item.get("business_model", ""),
                    item.get("segments", ""),
                    item.get("geography", ""),
                    item.get("therapeutic_focus", ""),
                    item.get("business_summary", ""),
                    company.company_name,
                )
            ).lower()
            groups = {
                "pharma": {"pharma", "pharmaceutical", "formulation", "generic", "api", "biolog"},
                "it": {"software", "technology", "cloud", "consult"},
                "energy": {"oil", "gas", "energy", "refin", "petrochemical", "upstream"},
                "metals": {"metal", "copper", "aluminium", "aluminum", "mining", "steel"},
                "auto": {"auto", "motor", "vehicle", "tractor", "mobility"},
                # Asset management is its own broad profile group so that AMCs are
                # not inadvertently pulled into the generic "banking" bucket.
                "asset_management": {"asset management", "amc", "mutual fund", "fund management"},
            }
            return {group for group, terms in groups.items() if any(term in raw for term in terms)}

        target_profile = broad_profile(target, target_evidence)
        target_revenue = target_evidence.get("revenue")
        ranked: List[Dict[str, object]] = []

        for candidate in candidates:
            item = evidence.get(candidate.symbol, {})
            candidate_summary = str(item.get("business_summary", ""))
            candidate_subtype = self.infer_financial_subtype(candidate, candidate_summary)
            candidate_profile = broad_profile(candidate, item)
            shared = target_profile.intersection(candidate_profile)

            score = 50.0
            reasons = [f"Same classified industry ({candidate.industry})"]

            # --- Financial sub-type similarity (only for Financial Services) -----
            if is_financial_industry:
                if target_subtype == candidate_subtype and target_subtype != "other_financial":
                    score += 40.0
                    reasons.append(
                        f"Matching financial sub-type: {target_subtype.replace('_', ' ')}"
                    )
                elif target_subtype != "other_financial" and candidate_subtype != "other_financial":
                    # Both have identifiable but different sub-types → strong penalty
                    score -= 35.0
                    reasons.append(
                        f"Different financial sub-types: target={target_subtype.replace('_', ' ')}, "
                        f"candidate={candidate_subtype.replace('_', ' ')} (provisional only)"
                    )
                else:
                    # At least one sub-type is unresolved
                    score -= 10.0
                    reasons.append(
                        f"Financial sub-type partially unresolved "
                        f"(target={target_subtype}, candidate={candidate_subtype}); provisional"
                    )
            # --- Broad business-profile groups (all industries) ----------------
            if target_profile and candidate_profile:
                if shared:
                    score += 20.0
                    reasons.append(
                        f"Shared business-profile evidence: {', '.join(sorted(shared))}"
                    )
                else:
                    score -= 15.0
                    reasons.append("No shared business-profile evidence")
            else:
                score -= 10.0
                reasons.append("Business-profile metadata unavailable; provisional")

            # --- Revenue-scale comparability ----------------------------------
            candidate_revenue = item.get("revenue")
            if isinstance(target_revenue, (int, float)) and isinstance(candidate_revenue, (int, float)):
                if target_revenue > 0 and candidate_revenue > 0:
                    scale_ratio = max(target_revenue, candidate_revenue) / min(target_revenue, candidate_revenue)
                    if scale_ratio <= 5:
                        score += 15.0
                        reasons.append("Revenue scale within 5x")
                    else:
                        score += 3.0
                        reasons.append("Revenue scale differs by more than 5x")
            else:
                reasons.append("Revenue-scale evidence unavailable")

            if item.get("has_financial_data"):
                score += 5.0
                reasons.append("Financial evidence available")

            if item.get("is_screener_peer"):
                score += 50.0
                reasons.append("Direct Screener.in peer comparison match")

            # Confidence is 'supported' only when sub-types match AND business
            # summary is available; otherwise the selection is 'provisional'.
            is_same_subtype = (
                is_financial_industry and target_subtype == candidate_subtype
                and target_subtype != "other_financial"
            )
            confidence = (
                "supported"
                if (is_same_subtype or (shared and not is_financial_industry))
                and item.get("business_summary")
                else "provisional"
            )

            ranked.append(
                {
                    "company": candidate,
                    "score": round(score, 2),
                    "reason": "; ".join(reasons),
                    "confidence": confidence,
                    "shared_business_profiles": sorted(shared),
                    "financial_subtype": candidate_subtype if is_financial_industry else None,
                }
            )
        ranked.sort(key=lambda row: (-float(row["score"]), row["company"].company_name))
        return ranked[:max_peers]

    @staticmethod
    def peer_selection_reason(
        target: CompanyRecord, peer: CompanyRecord
    ) -> str:
        if target.symbol in CURATED_PEER_GROUPS:
            curated_set = CURATED_PEER_GROUPS[target.symbol]
            if "NAM-INDIA" in (target.symbol, peer.symbol) or any(
                s in curated_set for s in ("ABSLAMC", "ICICIAMC", "HDFCAMC", "UTIAMC")
            ):
                return (
                    f"Curated listed asset-management peer for {target.company_name}; "
                    "both manage mutual funds in India. AUM, net inflows, and fee-rate "
                    "comparisons require segment disclosures and may not be available."
                )
            return (
                "Curated listed integrated-energy peer for company-wide Reliance "
                "comparison; segment-level conclusions require separate disclosures."
            )
        return (
            f"Same classified industry ({peer.industry}) as {target.company_name}; "
            "requires compatible annual evidence before comparison."
        )


_registry_instance: Optional[CompanyRegistry] = None


def get_company_registry() -> CompanyRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = CompanyRegistry()
    return _registry_instance
