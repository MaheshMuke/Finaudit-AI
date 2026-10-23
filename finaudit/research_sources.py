"""On-demand adapters for discovering company-specific primary documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol
from urllib.parse import urlencode

from finaudit.company_registry import CompanyRecord
from finaudit.schemas import Company


class PrimarySourceAdapter(Protocol):
    def discover(
        self,
        company: CompanyRecord,
        financial_year: str,
        statement_basis: str,
    ) -> dict[str, Any]: ...


@dataclass
class OfficialQuarterlyDisclosureAdapter:
    """Discovers relevant NSE announcement attachment candidates for one company."""

    fetch_announcements_fn: Callable[[CompanyRecord], tuple[Any, str, list[str]]] | None = None
    max_candidates: int = 8

    @staticmethod
    def _publication_date(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        for pattern in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:11], pattern).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None

EVENT_CATEGORIES = {
    "financial_results": (
        "quarterly result", "financial result", "financial statement", "earnings",
        "investor presentation", "results for", "audited result", "unaudited result",
        "press release - financial", "statement of profit",
    ),
    "dividend_or_corporate_action": (
        "dividend", "interim dividend", "final dividend", "bonus", "split",
        "sub-division", "buyback", "record date", "book closure", "rights issue",
    ),
    "board_meeting": (
        "board meeting", "outcome of board", "meeting of the board", "board to consider",
    ),
    "management_change": (
        "resignation", "appointment", "cessation", "key managerial", "kmp",
        "director", "auditor", "managing director", "ceo", "cfo",
    ),
    "acquisition_or_restructuring": (
        "acquisition", "amalgamation", "merger", "scheme of arrangement", "slump sale",
        "divestment", "joint venture", "subsidiary", "stake purchase", "demerger",
    ),
    "credit_rating": (
        "credit rating", "rating reaffirm", "rating upgrade", "rating downgrade",
        "crisil", "icra", "care rating", "india ratings",
    ),
    "regulatory_or_legal": (
        "order", "penalty", "show cause", "adjudication", "litigation", "dispute",
        "arbitration", "tax demand", "gst", "sebi", "cbi", "ed",
    ),
    "contracts_or_orders": (
        "order win", "contract", "bagged", "awarded", "agreement", "mou",
        "purchase order", "client win",
    ),
}


def classify_disclosure_event(title: str, text: str = "") -> str:
    combined = f"{title} {text}".lower()
    for cat, keywords in EVENT_CATEGORIES.items():
        if any(kw in combined for kw in keywords):
            return cat
    return "general_announcement"


def extract_disclosure_material_facts(title: str, text: str = "") -> dict[str, Any]:
    """Extract key monetary values, dates, percentages, and caveats from disclosure text."""
    combined = f"{title}\n{text}"
    facts: dict[str, Any] = {
        "event_type": classify_disclosure_event(title, text),
        "monetary_values": [],
        "percentages": [],
        "dates_mentioned": [],
        "conditions_and_caveats": [],
    }

    # Extract monetary values (e.g. Rs 1,500 Crore, $50 Million, Rs. 10 per share)
    money_matches = re.findall(
        r"(?:(?:Rs\.?|INR|USD|\$|₹)\s*[\d,]+(?:\.\d+)?\s*(?:crore|cr|lakh|million|mn|billion|bn|per share)?|\b[\d,]+(?:\.\d+)?\s*(?:crore|cr|lakh|million|billion)\b)",
        combined,
        re.I,
    )
    facts["monetary_values"] = list(dict.fromkeys(m.strip() for m in money_matches[:8]))

    # Extract percentages (e.g. 15.5%, 7%)
    pct_matches = re.findall(r"\b\d+(?:\.\d+)?%", combined)
    facts["percentages"] = list(dict.fromkeys(pct_matches[:6]))

    # Extract dates mentioned
    date_matches = re.findall(
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",
        combined,
        re.I,
    )
    facts["dates_mentioned"] = list(dict.fromkeys(date_matches[:5]))

    # Extract conditions and caveats
    caveat_indicators = [
        "subject to approval",
        "subject to shareholder",
        "subject to regulatory",
        "pending approval",
        "conditional upon",
        "without prejudice",
        "appeal filed",
        "stay granted",
        "disputed demand",
    ]
    for c in caveat_indicators:
        if c in combined.lower():
            facts["conditions_and_caveats"].append(c.title())

    return facts


@dataclass
class OfficialQuarterlyDisclosureAdapter:
    """Discovers, retrieves, and extracts relevant NSE announcements and attachments."""

    fetch_announcements_fn: Callable[[CompanyRecord], tuple[Any, str, list[str]]] | None = None
    max_candidates: int = 8
    max_response_bytes: int = 15 * 1024 * 1024

    @staticmethod
    def _publication_date(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        for pattern in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:11], pattern).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None

    def _fetch_nse_announcements(self, company: CompanyRecord) -> tuple[Any, str, list[str]]:
        import requests

        from finaudit.report_sources import _get_trusted, _http_session

        query = urlencode({"index": "equities", "symbol": company.symbol})
        endpoint = f"https://www.nseindia.com/api/corporate-announcements?{query}"
        try:
            response = _get_trusted(_http_session(), endpoint, company.symbol)
            try:
                if response.status_code != 200:
                    return None, endpoint, [f"NSE announcements returned HTTP {response.status_code}."]
                if len(response.content) > self.max_response_bytes:
                    return None, endpoint, [f"NSE announcement response exceeded the limit ({len(response.content)} bytes)."]
                return response.json(), endpoint, []
            finally:
                response.close()
        except requests.RequestException as exc:
            return None, endpoint, [f"NSE announcements unavailable ({type(exc).__name__})."]

    def discover(
        self,
        company: CompanyRecord,
        reporting_period: Any,
        statement_basis: str | None,
        from_date: str | None = None,
        to_date: str | None = None,
        event_types: list[str] | None = None,
    ) -> dict[str, Any]:
        fetch = self.fetch_announcements_fn or self._fetch_nse_announcements
        try:
            payload, endpoint, errors = fetch(company)
        except (OSError, TimeoutError, ValueError) as exc:
            return {
                "status": "unavailable",
                "reason": f"Official quarterly-disclosure discovery failed ({type(exc).__name__}).",
                "documents": [],
                "errors": [str(exc)],
            }
        if isinstance(payload, dict):
            entries = payload.get("data") or payload.get("announcements") or []
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []
        from finaudit.report_sources import is_trusted_url

        documents = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            title = str(
                item.get("desc")
                or item.get("subject")
                or item.get("title")
                or item.get("attachmentName")
                or ""
            ).strip()
            
            event_type = classify_disclosure_event(title)
            if event_types and event_type not in event_types:
                # If specific event types are requested, filter by them
                continue
            elif not event_types and not any(term in title.lower() for term in (
                "quarterly result", "financial result", "financial statement", "earnings",
                "investor presentation", "results for", "dividend", "board meeting",
                "acquisition", "order", "contract", "press release"
            )):
                continue

            pub_date = self._publication_date(
                item.get("an_dt") or item.get("broadcast_dttm") or item.get("date")
            )
            # Date range filtering if specified
            if from_date and pub_date and pub_date < from_date:
                continue
            if to_date and pub_date and pub_date > to_date:
                continue

            attachment = str(
                item.get("attchmntFile")
                or item.get("attachmentUrl")
                or item.get("attachment")
                or item.get("fileName")
                or ""
            ).strip()
            if not attachment or not is_trusted_url(attachment, company.symbol):
                continue
            item_basis = "unknown"
            normalized_title = title.lower().replace("-", " ")
            if "consolidated" in normalized_title and "standalone" not in normalized_title:
                item_basis = "consolidated"
            elif "standalone" in normalized_title or "stand alone" in normalized_title:
                item_basis = "standalone"
            if statement_basis in {"consolidated", "standalone"} and item_basis not in {
                "unknown",
                statement_basis,
            }:
                continue

            material_facts = extract_disclosure_material_facts(title)
            documents.append({
                "title": title or "NSE corporate disclosure",
                "source_kind": "nse_corporate_disclosure_candidate",
                "source_page": endpoint,
                "url": attachment,
                "publication_date": pub_date,
                "reporting_period": reporting_period,
                "statement_basis": item_basis if item_basis != "unknown" else statement_basis,
                "event_type": event_type,
                "material_facts": material_facts,
                "retrieved_at": datetime.now().astimezone().isoformat(),
                "content_retrieved": False,
            })
            if len(documents) >= self.max_candidates:
                break
        return {
            "status": "candidates_found" if documents else "not_found" if not errors else "unavailable",
            "reason": None if documents else "No relevant trusted quarterly-disclosure attachment was found.",
            "documents": documents,
            "errors": errors,
            "limitations": [
                "Discovered attachments require explicit extraction before content analysis.",
                "Candidate links indicate a disclosure was filed but are not evidence for a causal explanation until the text is extracted and verified.",
            ],
        }

    def download_and_extract_attachment(
        self,
        document: dict[str, Any],
        company: CompanyRecord,
        timeout: int = 20,
    ) -> dict[str, Any]:
        """Download announcement attachment, compute SHA-256 hash, and extract clean text and facts."""
        import hashlib
        from finaudit.report_sources import _http_session, is_trusted_url

        url = document.get("url")
        if not url or not is_trusted_url(url, company.symbol):
            return {
                "status": "failed",
                "error": f"Untrusted or missing attachment URL: {url}",
                "extracted_text": "",
                "checksum": None,
            }

        session = _http_session()
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code != 200:
                return {
                    "status": "failed",
                    "error": f"Attachment returned HTTP {resp.status_code}",
                    "extracted_text": "",
                    "checksum": None,
                }

            content = resp.content
            checksum = hashlib.sha256(content).hexdigest()
            extracted_text = ""
            is_pdf = content.startswith(b"%PDF-") or url.lower().endswith(".pdf")

            if is_pdf:
                try:
                    import fitz
                    doc = fitz.open(stream=content, filetype="pdf")
                    pages_text = []
                    for i in range(min(len(doc), 15)):
                        pt = doc[i].get_text()
                        if pt.strip():
                            pages_text.append(f"--- Page {i+1} ---\n{pt.strip()}")
                    extracted_text = "\n\n".join(pages_text)
                except Exception as pdf_err:
                    extracted_text = f"[PDF extraction fallback: {pdf_err}]"
            else:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, "html.parser")
                extracted_text = soup.get_text(separator="\n", strip=True)

            material_facts = extract_disclosure_material_facts(
                document.get("title", ""), extracted_text
            )

            result = {
                "status": "extracted",
                "url": url,
                "title": document.get("title", ""),
                "event_type": document.get("event_type") or classify_disclosure_event(document.get("title", "")),
                "publication_date": document.get("publication_date"),
                "checksum": checksum,
                "byte_size": len(content),
                "extracted_text": extracted_text[:8000],
                "material_facts": material_facts,
                "retrieved_at": datetime.now().astimezone().isoformat(),
            }
            return result
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"Failed downloading attachment: {type(exc).__name__}: {exc}",
                "extracted_text": "",
                "checksum": None,
            }

    def discover_multi(
        self,
        companies: list[Any],
        from_date: str | None = None,
        to_date: str | None = None,
        event_types: list[str] | None = None,
        extract_content: bool = False,
    ) -> dict[str, Any]:
        """Multi-company disclosure intelligence with date filtering and coverage tracking."""
        from finaudit.company_registry import get_company_registry, CompanyRecord

        registry = get_company_registry()
        norm_companies: list[CompanyRecord] = []
        unresolved: list[str] = []

        for c in companies:
            if isinstance(c, str):
                rec = registry.get_by_symbol(c.strip().upper())
                if rec:
                    norm_companies.append(rec)
                else:
                    norm_companies.append(
                        CompanyRecord(
                            symbol=c.strip().upper(),
                            company_name=c.strip().upper(),
                            industry="Unknown",
                            isin="",
                            yahoo_ticker=f"{c.strip().upper()}.NS",
                        )
                    )
            elif isinstance(c, CompanyRecord):
                norm_companies.append(c)

        coverage: dict[str, Any] = {
            "companies_queried": [c.symbol for c in norm_companies],
            "total_companies": len(norm_companies),
            "results_by_company": {},
            "all_documents": [],
            "documents": [],
            "coverage_stats": {"success": 0, "empty": 0, "failed": 0},
            "failures": [],
        }

        for comp in norm_companies:
            res = self.discover(
                comp,
                reporting_period=None,
                statement_basis=None,
                from_date=from_date,
                to_date=to_date,
                event_types=event_types,
            )
            docs = res.get("documents", [])
            status = res.get("status")
            if extract_content and docs:
                for doc in docs[:2]:
                    extracted = self.download_and_extract_attachment(doc, comp)
                    if extracted.get("status") == "extracted":
                        doc["content_retrieved"] = True
                        doc["checksum"] = extracted.get("checksum")
                        doc["extracted_text"] = extracted.get("extracted_text")
                        doc["material_facts"] = extracted.get("material_facts")

            if status == "candidates_found":
                coverage["coverage_stats"]["success"] += 1
            elif status == "not_found":
                coverage["coverage_stats"]["empty"] += 1
            else:
                coverage["coverage_stats"]["failed"] += 1
                coverage["failures"].append({"symbol": comp.symbol, "errors": res.get("errors", [])})

            coverage["results_by_company"][comp.symbol] = {
                "symbol": comp.symbol,
                "company_name": comp.company_name,
                "status": "success" if status == "candidates_found" else status,
                "documents_count": len(docs),
                "documents": docs,
                "errors": res.get("errors", []),
            }
            coverage["all_documents"].extend(docs)

        coverage["documents"] = coverage["all_documents"]
        coverage["total_documents"] = len(coverage["documents"])
        coverage["coverage"] = coverage["results_by_company"]
        coverage["status"] = "completed" if coverage["coverage_stats"]["failed"] == 0 else "partial"
        return coverage



@dataclass
class OfficialAnnualReportAdapter:
    """Uses the existing trusted-domain report discovery without downloading documents."""

    discover_candidates_fn: Callable[..., tuple[list[dict[str, Any]], list[str]]] | None = None

    def discover(
        self,
        company: CompanyRecord,
        financial_year: str,
        statement_basis: str,
    ) -> dict[str, Any]:
        if statement_basis not in {"consolidated", "standalone"}:
            return {
                "status": "unavailable",
                "reason": "A single consolidated or standalone reporting basis is required.",
                "documents": [],
                "errors": [],
            }
        discover_candidates = self.discover_candidates_fn
        if discover_candidates is None:
            from finaudit.report_sources import discover_candidates

        try:
            candidates, errors = discover_candidates(
                Company(
                    company_id=company.symbol,
                    name=company.company_name,
                    symbol=company.symbol,
                    sector=company.industry or None,
                ),
                financial_year,
                statement_basis,
            )
        except (ValueError, OSError, TimeoutError) as exc:
            return {
                "status": "unavailable",
                "reason": f"Official annual-report discovery failed ({type(exc).__name__}).",
                "documents": [],
                "errors": [str(exc)],
            }
        documents = [
            {
                "title": item.get("title"),
                "source_kind": item.get("source_kind"),
                "source_page": item.get("source_page"),
                "url": item.get("pdf_url"),
                "publication_date": item.get("publication_date"),
                "financial_year": item.get("financial_year"),
                "statement_basis": item.get("statement_basis"),
                "retrieved_at": datetime.now().astimezone().isoformat(),
                "content_retrieved": False,
            }
            for item in candidates
            if item.get("pdf_url")
        ]
        return {
            "status": "candidates_found" if documents else "not_found",
            "reason": None if documents else "No trusted annual-report candidate was found.",
            "documents": documents,
            "errors": errors,
            "limitations": [
                "Discovery returns annual-report links only; it does not retrieve or verify report contents.",
                "Annual reports are not substitutes for quarterly exchange disclosures or investor presentations.",
            ],
        }


def financial_year_for_period(period: Any) -> str | None:
    try:
        parsed = datetime.strptime(str(period)[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None
    return f"FY{parsed.year + (1 if parsed.month >= 4 else 0)}"
