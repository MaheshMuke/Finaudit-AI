import hashlib
import html
import json
import logging
import os
import re
import email.utils
import tempfile
import time
from datetime import date, datetime
import uuid
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import fitz
import requests

from finaudit import database
from finaudit.llm import (
    configured_model_name,
    create_llm_client,
)
from finaudit.schemas import Company

logger = logging.getLogger(__name__)

MAX_REPORT_BYTES = 25 * 1024 * 1024
REQUEST_TIMEOUT = (5, 20)
MAX_RETRIES = 2
OFFICIAL_DOMAINS = {
    "INFY": ("infosys.com",),
    "TCS": ("tcs.com",),
    "WIPRO": ("wipro.com",),
    "HCLTECH": ("hcltech.com",),
    "TECHM": ("techmahindra.com",),
    "RELIANCE": ("ril.com",),
    "ICICIBANK": ("icicibank.com",),
    "NTPC": ("ntpc.co.in",),
    "GRASIM": ("grasim.com",),
}
EXCHANGE_DOMAINS = ("nseindia.com", "nsearchives.nseindia.com", "bseindia.com")
BSE_CODES = {
    "INFY": "500209",
    "TCS": "532540",
    "WIPRO": "507685",
    "HCLTECH": "532281",
    "TECHM": "532755",
    "RELIANCE": "500325",
    "ICICIBANK": "532174",
}

# This exact official URL is present in the Infosys sample report itself.
VERIFIED_OFFICIAL_LINKS = {
    ("INFY", "FY2026"): {
        "title": "Infosys Integrated Annual Report 2025-26",
        "source_page": "https://www.infosys.com/investors/reports-filings/annual-2025-26",
        "pdf_url": (
            "https://www.infosys.com/investors/reports-filings/"
            "annual-report/annual/documents/infosys-ar-26.pdf"
        ),
        "publication_date": "2026-05-29",
    }
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.active_index: int | None = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        values = dict(attrs)
        href = values.get("href")
        if href:
            self.links.append((href, ""))
            self.active_index = len(self.links) - 1

    def handle_data(self, data):
        if self.active_index is not None:
            href, label = self.links[self.active_index]
            self.links[self.active_index] = (href, f"{label} {data}".strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a":
            self.active_index = None


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(text).lower()).strip()


def official_domains(company_id: str) -> tuple[str, ...]:
    return OFFICIAL_DOMAINS.get(company_id, ()) + EXCHANGE_DOMAINS


def is_trusted_url(url: str, company_id: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return False
        host = parsed.hostname.lower().rstrip(".")
        return any(host == domain or host.endswith("." + domain) for domain in official_domains(company_id))
    except (TypeError, ValueError):
        return False


def _normalize_fy(year: str) -> str:
    year_str = year.strip()
    match_range = re.fullmatch(r"(?:FY)?(20\d{2})[-/](\d{2,4})", year_str, re.I)
    if match_range:
        start_year = int(match_range.group(1))
        end_part = match_range.group(2)
        end_year = int(str(start_year)[:2] + end_part) if len(end_part) == 2 else int(end_part)
        return f"FY{end_year}"
    match_single = re.fullmatch(r"(?:FY)?(20\d{2})", year_str, re.I)
    if match_single:
        return f"FY{match_single.group(1)}"
    raise ValueError("Financial year must look like FY2026 or 2025-26.")


def _basis_from_text(text: str) -> str:
    normalized = _normal(text)
    consolidated = "consolidated" in normalized
    standalone = "standalone" in normalized or "stand alone" in normalized
    if consolidated and not standalone:
        return "consolidated"
    if standalone and not consolidated:
        return "standalone"
    return "unknown"


def _candidate(
    company: Company,
    financial_year: str,
    statement_basis: str,
    title: str,
    source_page: str,
    pdf_url: str,
    source_kind: str,
    publication_date: str | None = None,
) -> dict:
    if isinstance(publication_date, str) and publication_date:
        try:
            publication_date = datetime.fromisoformat(
                publication_date.replace("Z", "+00:00")
            ).date().isoformat()
        except ValueError:
            publication_date = None
    else:
        publication_date = None
    return {
        "candidate_id": str(uuid.uuid4()),
        "company_id": company.company_id,
        "company_name": company.name,
        "symbol": company.symbol,
        "financial_year": financial_year,
        "statement_basis": _basis_from_text(f"{title} {source_page}"),
        "requested_basis": statement_basis,
        "publication_date": publication_date,
        "title": title.strip() or "Annual report PDF",
        "source_site": urlparse(source_page).hostname or "",
        "source_page": source_page,
        "pdf_url": pdf_url,
        "source_kind": source_kind,
        "download_status": "candidate",
        "report_id": None,
        "verification_results": None,
        "error": None,
    }


def _http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
    )
    return session


def _get_with_retries(session: requests.Session, url: str, **kwargs):
    kwargs.setdefault("allow_redirects", False)
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt == MAX_RETRIES:
                return response
            retry_after = response.headers.get("retry-after")
            delay = 0.25 * (attempt + 1)
            if retry_after:
                try:
                    delay = min(max(float(retry_after), 0), 2)
                except ValueError:
                    parsed_date = email.utils.parsedate_to_datetime(retry_after)
                    delay = min(
                        max((parsed_date - datetime.now(parsed_date.tzinfo)).total_seconds(), 0),
                        2,
                    )
            response.close()
            time.sleep(delay)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(0.25 * (attempt + 1))
    if last_error:
        raise last_error
    raise requests.RequestException(f"Could not retrieve {url}")


def _get_trusted(
    session: requests.Session,
    url: str,
    company_id: str,
    **kwargs,
):
    current_url = url
    for _ in range(4):
        if not is_trusted_url(current_url, company_id):
            raise ValueError("Official source redirected outside approved domains.")
        response = _get_with_retries(session, current_url, **kwargs)
        if not is_trusted_url(getattr(response, "url", current_url), company_id):
            response.close()
            raise ValueError("Official source response resolved outside approved domains.")
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("location")
        response.close()
        if not location:
            raise requests.HTTPError("Official source returned a redirect without a location.")
        current_url = urljoin(current_url, location)
    raise requests.TooManyRedirects("Official source exceeded the redirect limit.")


def _fy_matches(text: str, financial_year: str) -> bool:
    year = int(financial_year[2:])
    normalized = _normal(text)
    previous_two = f"{year - 1} {str(year)[-2:]}"
    patterns = [
        previous_two,
        f"{year - 1} {year}",
        f"fy {year}",
        f"fy {str(year)[-2:]}",
        f"financial year {year}",
        f"financial year {year - 1} {year}",
        f"financial year {year - 1} {str(year)[-2:]}",
        f"annual report {year - 1} {str(year)[-2:]}",
        f"annual report {year - 1} {year}",
        f"annual report {year}",
        f"report {year - 1} {str(year)[-2:]}",
        f"report {year - 1} {year}",
        f"report {year}",
        f"year ended march 31 {year}",
        f"ended march 31 {year}",
        f"as on march 31 {year}",
        f"as at march 31 {year}",
        f"31 03 {year}",
        f"31 3 {year}",
    ]
    if any(pattern in normalized for pattern in patterns):
        return True
    # Word-boundary check for explicit fiscal year like "2025-26" or "2025-2026"
    if re.search(rf"\b{year - 1}[-/]{str(year)[-2:]}\b", text, re.I):
        return True
    if re.search(rf"\b{year - 1}[-/]{year}\b", text, re.I):
        return True
    if re.search(rf"\bFY\s*{year}\b", text, re.I):
        return True
    return False


def _publication_date_in_pdf(text: str) -> str | None:
    patterns = (
        (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b", "%B %d %Y"),
        (r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b", "%d %B %Y"),
    )
    for pattern, date_format in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value = " ".join(match.groups())
            try:
                return datetime.strptime(value.title(), date_format).date().isoformat()
            except ValueError:
                continue
    for match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text):
        try:
            return date.fromisoformat(match.group()).isoformat()
        except ValueError:
            continue
    return None


class OpenAIWebSearchProvider:
    """Search adapter using OpenAI's Responses API hosted web_search tool."""

    def __init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        self.client = OpenAI(timeout=35, max_retries=1)
        self.model = os.getenv("OPENAI_SEARCH_MODEL", "gpt-5.5")

    def search(self, company: Company, financial_year: str, statement_basis: str) -> list[dict]:
        domains = list(official_domains(company.company_id))
        response = self.client.responses.create(
            model=self.model,
            tools=[
                {
                    "type": "web_search",
                    "filters": {"allowed_domains": domains},
                }
            ],
            max_output_tokens=1400,
            input=(
                "Find official annual report PDFs only. Return JSON with a candidates array; "
                "each candidate must have title, direct_pdf_url, source_page_url, publication_date "
                "(ISO date or null), and evidence_excerpt. Do not invent URLs. Search only official "
                "company or exchange domains. The requested company is "
                f"{company.name} (symbol {company.symbol or company.company_id}), financial year "
                f"{financial_year}, requested statements {statement_basis}. "
                "Prefer investor relations report sources, then NSE/BSE disclosure sources. "
                "Return only direct PDF links surfaced by search results, not guessed URLs."
            ),
        )
        return _search_candidates(company, financial_year, statement_basis, response.output_text)


class OpenRouterWebSearchProvider:
    """Search adapter using OpenRouter's optional hosted web-search plugin."""

    def __init__(self) -> None:
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        self.client = create_llm_client(timeout=35)
        self.model = os.getenv("OPENROUTER_SEARCH_MODEL") or os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

    def search(self, company: Company, financial_year: str, statement_basis: str) -> list[dict]:
        domains = list(official_domains(company.company_id))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1400,
                extra_body={"plugins": [{"id": "web", "max_results": 5}]},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Find official annual report PDFs only. Return JSON with a candidates "
                            "array; each candidate must have title, direct_pdf_url, source_page_url, "
                            "publication_date (ISO date or null), and evidence_excerpt. Do not invent "
                            "URLs. Search only these allowed official domains: "
                            f"{', '.join(domains)}. Prefer issuer "
                            "investor-relations pages, then exchange disclosures. Return direct PDF "
                            "links actually found in search results, not guessed URLs."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Find the official {company.name} (symbol "
                            f"{company.symbol or company.company_id}) annual report for "
                            f"{financial_year}, reporting basis {statement_basis}."
                        ),
                    },
                ],
            )
            message = response.choices[0].message
            output = message.content or ""
            return _search_candidates(
                company,
                financial_year,
                statement_basis,
                output,
                source_kind="openrouter_web_search",
            )
        except Exception as exc:
            logger.warning(
                "OpenRouter web search provider failed (%s): %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            raise


def _search_candidates(
    company: Company,
    financial_year: str,
    statement_basis: str,
    output: str,
    source_kind: str = "openai_responses_web_search",
) -> list[dict]:
    json_match = re.search(r"\{.*\}", output, flags=re.S)
    if not json_match:
        logger.warning("LLM report search returned no structured candidates.")
        return []
    try:
        payload = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("LLM report search returned malformed candidate JSON.")
        return []
    candidates = []
    if not isinstance(payload, dict):
        return []
    for item in payload.get("candidates", []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("direct_pdf_url") or "").strip()
        source_page = str(item.get("source_page_url") or url).strip()
        if not is_trusted_url(url, company.company_id) or not is_trusted_url(
            source_page, company.company_id
        ):
            continue
        candidates.append(
            _candidate(
                company,
                financial_year,
                statement_basis,
                str(item.get("title") or ""),
                source_page,
                url,
                source_kind,
                item.get("publication_date"),
            )
        )
    return candidates


def _html_candidates(
    company: Company,
    financial_year: str,
    statement_basis: str,
    session: requests.Session,
) -> tuple[list[dict], list[str]]:
    domains = OFFICIAL_DOMAINS.get(company.company_id, ())
    if not domains:
        return [], ["No verified company investor-relations domain is configured."]
    errors = []
    found = []
    pages = [f"https://www.{domains[0]}/"]
    if company.company_id == "INFY":
        pages.append("https://www.infosys.com/investors/reports-filings/annual-2025-26")
    visited = set()
    page_index = 0
    while page_index < len(pages) and page_index < 6:
        page_url = pages[page_index]
        page_index += 1
        if page_url in visited:
            continue
        visited.add(page_url)
        if not is_trusted_url(page_url, company.company_id):
            continue
        try:
            response = _get_trusted(session, page_url, company.company_id)
            if response.status_code >= 400:
                errors.append(f"Official company page returned HTTP {response.status_code}: {page_url}")
                continue
            if "html" not in response.headers.get("content-type", "").lower():
                continue
            parser = LinkParser()
            parser.feed(response.text[:2_000_000])
            for href, label in parser.links:
                link = urljoin(response.url, html.unescape(href))
                if not is_trusted_url(link, company.company_id):
                    continue
                searchable = f"{label} {link}"
                if ".pdf" in urlparse(link).path.lower():
                    if not _fy_matches(searchable, financial_year):
                        continue
                    if _basis_from_text(searchable) not in {"unknown", statement_basis}:
                        continue
                    found.append(
                        _candidate(
                            company,
                            financial_year,
                            statement_basis,
                            label or Path(urlparse(link).path).name,
                            response.url,
                            link,
                            "company_ir_crawl",
                        )
                    )
                elif (
                    page_index < 6
                    and any(word in searchable.lower() for word in ("investor", "annual", "report"))
                    and link not in visited
                    and link not in pages
                ):
                    pages.append(link)
        except requests.RequestException as exc:
            errors.append(f"Official company page unavailable ({type(exc).__name__}): {page_url}")
    return found, errors


def _exchange_candidates(
    company: Company,
    financial_year: str,
    statement_basis: str,
    session: requests.Session,
) -> tuple[list[dict], list[str]]:
    symbol = company.symbol or company.company_id
    found = []
    errors = []

    # 1. Query official NSE filings endpoint
    target_end_year = int(financial_year[2:])
    nse_api_url = f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
    try:
        api_response = _get_trusted(session, nse_api_url, company.company_id)
        if api_response.status_code == 200:
            try:
                data = api_response.json().get("data", [])
                for item in data:
                    to_yr = str(item.get("toYr", "")).strip()
                    file_name = str(item.get("fileName", "")).strip()
                    if to_yr == str(target_end_year) and file_name.lower().endswith(".pdf"):
                        if is_trusted_url(file_name, company.company_id):
                            pub_dt = item.get("broadcast_dttm") or item.get("disseminationDateTime")
                            pub_iso = None
                            if pub_dt:
                                try:
                                    dt_match = re.search(r"(\d{2}-[A-Za-z]{3}-\d{4})", pub_dt)
                                    if dt_match:
                                        pub_iso = datetime.strptime(dt_match.group(1), "%d-%b-%Y").date().isoformat()
                                except Exception:
                                    pub_iso = None
                            found.append(
                                _candidate(
                                    company,
                                    financial_year,
                                    statement_basis,
                                    f"{company.name} Annual Report {item.get('fromYr', target_end_year - 1)}-{to_yr}",
                                    nse_api_url,
                                    file_name,
                                    "nse_corporate_disclosure",
                                    pub_iso,
                                )
                            )
            except Exception as json_err:
                logger.debug("Failed parsing NSE annual reports JSON: %s", json_err)
        elif api_response.status_code >= 400:
            errors.append(f"NSE official filings API returned HTTP {api_response.status_code}.")
    except requests.RequestException as exc:
        errors.append(f"NSE official filings API unavailable ({type(exc).__name__}).")

    # 2. Query HTML exchange listings
    pages = [
        (
            "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports"
            f"?symbol={symbol}&tabIndex=equity",
            "NSE",
        )
    ]
    bse_code = BSE_CODES.get(company.company_id)
    if bse_code:
        pages.append((f"https://www.bseindia.com/corporates/ann.html?Scrip={bse_code}", "BSE"))
    for page_url, name in pages:
        try:
            response = _get_trusted(session, page_url, company.company_id)
            if response.status_code >= 400:
                errors.append(f"{name} official filings page returned HTTP {response.status_code}.")
                continue
            parser = LinkParser()
            parser.feed(response.text[:2_000_000])
            for href, label in parser.links:
                link = urljoin(response.url, html.unescape(href))
                if not is_trusted_url(link, company.company_id):
                    continue
                if ".pdf" not in urlparse(link).path.lower():
                    continue
                searchable = f"{label} {link}"
                if not _fy_matches(searchable, financial_year):
                    continue
                found.append(
                    _candidate(
                        company,
                        financial_year,
                        statement_basis,
                        label or Path(urlparse(link).path).name,
                        response.url,
                        link,
                        f"{name.lower()}_corporate_disclosure",
                    )
                )
        except requests.RequestException as exc:
            errors.append(f"{name} official filings unavailable ({type(exc).__name__}).")
    return found, errors


def discover_candidates(
    company: Company,
    financial_year: str,
    statement_basis: str,
    search_provider=None,
    session: requests.Session | None = None,
) -> tuple[list[dict], list[str]]:
    year = _normalize_fy(financial_year)
    if statement_basis not in {"consolidated", "standalone"}:
        raise ValueError("Statement basis must be consolidated or standalone.")
    session = session or _http_session()
    candidates = []
    errors = []
    verified_seed = VERIFIED_OFFICIAL_LINKS.get((company.company_id, year))
    if verified_seed:
        candidates.append(
            _candidate(
                company,
                year,
                statement_basis,
                verified_seed["title"],
                verified_seed["source_page"],
                verified_seed["pdf_url"],
                "verified_company_ir_link",
                verified_seed["publication_date"],
            )
        )
    ir_found, ir_errors = _html_candidates(company, year, statement_basis, session)
    candidates.extend(ir_found)
    errors.extend(ir_errors)
    exchange_found, exchange_errors = _exchange_candidates(
        company, year, statement_basis, session
    )
    candidates.extend(exchange_found)
    errors.extend(exchange_errors)
    if search_provider:
        try:
            candidates.extend(search_provider.search(company, year, statement_basis))
        except Exception as exc:
            logger.warning(
                "Optional web search failed (%s): %s", type(exc).__name__, exc, exc_info=True
            )
            errors.append(f"Optional web search unavailable ({type(exc).__name__}: {exc}).")
    elif os.getenv("OPENROUTER_API_KEY"):
        try:
            candidates.extend(
                OpenRouterWebSearchProvider().search(
                    company, year, statement_basis
                )
            )
        except Exception as exc:
            logger.warning(
                "Optional OpenRouter web search failed (%s): %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            errors.append(
                f"Optional OpenRouter web search unavailable ({type(exc).__name__}: {exc})."
            )
    elif os.getenv("OPENAI_API_KEY"):
        try:
            candidates.extend(
                OpenAIWebSearchProvider().search(company, year, statement_basis)
            )
        except Exception as exc:
            logger.warning(
                "Optional web search failed (%s): %s", type(exc).__name__, exc, exc_info=True
            )
            errors.append(f"Optional web search unavailable ({type(exc).__name__}: {exc}).")
    else:
        errors.append(
            "Optional LLM web search is not configured; official-source crawling was attempted."
        )

    unique = {}
    for item in candidates:
        if is_trusted_url(item["pdf_url"], company.company_id):
            unique.setdefault(item["pdf_url"], item)
    return list(unique.values()), errors


def download_and_verify(
    candidate: dict,
    company: Company,
    financial_year: str,
    statement_basis: str,
    session: requests.Session | None = None,
) -> dict:
    year = _normalize_fy(financial_year)
    url = candidate.get("pdf_url", "")
    if not is_trusted_url(url, company.company_id):
        raise ValueError("Candidate PDF URL is not HTTPS on an approved official domain.")
    session = session or _http_session()
    response = _get_trusted(
        session,
        url,
        company.company_id,
        stream=True,
    )
    if response.status_code != 200:
        raise requests.HTTPError(f"Official PDF request returned HTTP {response.status_code}.")
    final_url = getattr(response, "url", url)
    if not is_trusted_url(final_url, company.company_id):
        raise ValueError("The PDF URL redirected outside the approved official domains.")
    content_type = response.headers.get("content-type", "").lower()
    disposition = response.headers.get("content-disposition", "").lower()
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_REPORT_BYTES:
        response.close()
        raise ValueError("Annual report exceeds the 25 MB download limit.")
    if "text/html" in content_type and "application/pdf" not in content_type:
        raise ValueError("Official source returned an HTML page, not a PDF.")
    target_dir = database.REPORTS_DIR / company.company_id / year
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    digest = hashlib.sha256()
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".download-", suffix=".part", dir=target_dir, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_REPORT_BYTES:
                    raise ValueError("Annual report exceeds the 25 MB download limit.")
                temporary.write(chunk)
                digest.update(chunk)
        if total < 8 or temp_path.read_bytes()[:5] != b"%PDF-":
            if "text/html" in content_type or temp_path.read_bytes().lstrip().lower().startswith(
                (b"<!doctype html", b"<html")
            ):
                raise ValueError("Official source returned an HTML error page, not a PDF.")
            raise ValueError("Downloaded document does not have a valid PDF signature.")
        checksum = digest.hexdigest()
        existing = database.get_verified_report(
            company.company_id, year, statement_basis
        )
        if existing and existing["checksum"] == checksum:
            return {
                "local_path": existing["local_path"],
                "checksum": checksum,
                "verification_results": json.loads(existing["verification_json"] or "{}"),
                "reused_report": existing,
                "final_url": final_url,
                "publication_date": candidate.get("publication_date"),
            }
        try:
            document = fitz.open(stream=temp_path.read_bytes(), filetype="pdf")
        except Exception as exc:
            raise ValueError("Downloaded file is not an openable PDF document.") from exc
        if document.page_count < 3:
            document.close()
            raise ValueError("Downloaded PDF is too short to be an annual report.")
        page_count = document.page_count
        text_parts = [
            document[index].get_text("text")
            for index in range(min(page_count, 60))
        ]
        document.close()
        text = "\n".join(text_parts)
        normalized_text = _normal(text)
        identity_tokens = [
            token
            for token in _normal(company.name).split()
            if token not in {"limited", "ltd", "company", "co", "the", "india"}
        ]
        company_match = any(
            len(token) >= 3 and re.search(rf"\b{re.escape(token)}\b", normalized_text)
            for token in identity_tokens
        )
        symbol_match = bool(
            company.symbol
            and re.search(rf"\b{re.escape(_normal(company.symbol))}\b", normalized_text)
        )
        if not company_match and not symbol_match:
            raise ValueError(
                "PDF does not contain the selected company name or symbol in its opening pages."
            )
        if not _fy_matches(text, year):
            raise ValueError(
                f"PDF does not contain evidence for the requested financial year {year}."
            )
        if "annual report" not in normalized_text:
            raise ValueError("PDF content does not identify itself as an annual report.")
        from finaudit.extraction import extract_pdf

        extraction = extract_pdf(temp_path, statement_basis)
        if year not in extraction["financial_years"]:
            raise ValueError(
                f"Could not verify {statement_basis} statements for {year} in the PDF."
            )
        destination = target_dir / f"annual-report-{statement_basis}-{checksum[:16]}.pdf"
        temp_path.replace(destination)
        verification = {
            "status": "verified",
            "pdf_signature": True,
            "pdf_openable": True,
            "company_identity": "matched" if company_match else "symbol matched",
            "financial_year": year,
            "statement_basis": statement_basis,
            "extracted_financial_years": extraction["financial_years"],
            "page_count": page_count,
            "content_type": content_type,
            "content_disposition": disposition,
            "bytes": total,
            "official_source_url": final_url,
            "source_provenance": {
                "source_page": candidate.get("source_page"),
                "pdf_url": candidate.get("pdf_url"),
                "source_kind": candidate.get("source_kind"),
            },
            "publication_date_in_pdf": _publication_date_in_pdf(text),
            "candidate_publication_date_appears_in_pdf": bool(
                candidate.get("publication_date")
                and candidate["publication_date"] in text
            ),
            "report_title_evidence": next(
                (
                    line.strip()[:240]
                    for line in text.splitlines()
                    if "annual report" in line.lower()
                ),
                None,
            ),
        }
        return {
            "local_path": str(destination),
            "checksum": checksum,
            "verification_results": verification,
            "reused_report": None,
            "final_url": final_url,
            "publication_date": candidate.get("publication_date"),
            "extraction": extraction,
        }
    except Exception:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
    finally:
        response.close()
