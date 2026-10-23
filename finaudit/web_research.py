"""Practical, evidence-first web research service for FinAudit AI.

Discovers, retrieves, and extracts company-specific information from public web pages,
official filings, and PDF documents with caching, deduplication, and entity relevance filtering.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from finaudit.news_service import NewsService

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


FINANCIAL_STOP_WORDS = {
    "why", "did", "does", "do", "how", "what", "is", "are", "was", "were", "has", "have", "had",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "about", "into", "over", "after", "company", "limited", "ltd", "corp", "corporation", "inc",
    "please", "tell", "me", "explain", "analyze", "show", "give", "can", "could", "would", "should",
    "despite", "although", "while", "which", "where", "who", "whom", "whose", "this", "that", "these",
    "statement", "financial", "performance", "numbers", "report", "question",
}


def clean_text(raw: str) -> str:
    """Normalize whitespace and strip control characters."""
    if not raw:
        return ""
    return " ".join(raw.split()).strip()


def extract_issue_keywords(question: str, company_name: str = "", symbol: str = "") -> List[str]:
    """Extract financial issue keywords (e.g. 'profit', 'fall', 'revenue', 'growth') from question."""
    if not question:
        return []
    words = re.findall(r"[A-Za-z0-9\-]+", question)
    exclude = set(FINANCIAL_STOP_WORDS)
    for part in (company_name or "").split():
        exclude.add(part.lower())
    for part in (symbol or "").split("-"):
        exclude.add(part.lower())
    exclude.add((symbol or "").lower())

    keywords = []
    for w in words:
        wl = w.lower()
        if wl not in exclude and len(wl) >= 3 and not wl.isdigit():
            if wl not in [k.lower() for k in keywords]:
                keywords.append(w)
    return keywords


def matches_entity(text: str, entities: List[str]) -> bool:
    """Check if text contains any of the target entities with word-boundary safety."""
    if not entities:
        return True
    for entity in entities:
        ent = str(entity).strip()
        if not ent:
            continue
        if len(ent) <= 4:
            if re.search(rf"\b{re.escape(ent)}\b", text, re.IGNORECASE):
                return True
        else:
            if ent.lower() in text.lower():
                return True
    return False


class WebResearchService:
    """Practical web retrieval and document extraction service."""

    def __init__(self, news_service: Optional[NewsService] = None):
        self.news_service = news_service or NewsService()
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        self._doc_cache: Dict[str, Dict[str, Any]] = {}
        self._search_cache: Dict[str, List[Dict[str, Any]]] = {}

    def search_web(
        self,
        query: str,
        limit: int = 5,
        required_entities: Optional[List[str]] = None,
        timeout: int = 8,
    ) -> List[Dict[str, Any]]:
        """Search public web for relevant pages and PDF disclosures.
        
        Uses DuckDuckGo HTML search as primary live search, with fallback to local/RSS news.
        Filters by entity to reject irrelevant results and deduplicates by URL and title.
        """
        cache_key = f"{query.strip().lower()}:{limit}"
        if cache_key in self._search_cache:
            cached = self._search_cache[cache_key]
            if required_entities:
                return [r for r in cached if matches_entity(f"{r['title']} {r.get('snippet', '')}", required_entities)]
            return cached

        results: List[Dict[str, Any]] = []
        seen_links = set()
        seen_titles = set()

        diagnostics: Dict[str, Any] = {"query": query, "providers_attempted": []}

        # 1. Try Live Search (DuckDuckGo HTML)
        try:
            diagnostics["providers_attempted"].append("duckduckgo_html")
            resp = self.session.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                timeout=timeout,
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for result in soup.find_all("div", class_="result"):
                    heading = result.find("h2", class_="result__title")
                    snippet_elem = result.find("a", class_="result__snippet")
                    if not heading:
                        continue
                    a_tag = heading.find("a")
                    if not a_tag:
                        continue

                    raw_href = a_tag.get("href", "")
                    link = raw_href
                    if "uddg=" in raw_href:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                        link = parsed.get("uddg", [raw_href])[0]

                    if not link.startswith(("http://", "https://")):
                        continue

                    title = clean_text(a_tag.get_text())
                    snippet = clean_text(snippet_elem.get_text()) if snippet_elem else ""

                    combined_text = f"{title} {snippet}"
                    if required_entities and not matches_entity(combined_text, required_entities):
                        continue

                    norm_title = "".join(c.lower() for c in title if c.isalnum())
                    if link in seen_links or norm_title in seen_titles:
                        continue

                    seen_links.add(link)
                    if norm_title:
                        seen_titles.add(norm_title)

                    domain = urllib.parse.urlparse(link).netloc.replace("www.", "")
                    results.append({
                        "title": title,
                        "link": link,
                        "source": domain or "Web Search",
                        "summary": snippet,
                        "snippet": snippet,
                        "published": None,
                        "publication_date": None,
                        "is_pdf": link.lower().endswith(".pdf"),
                    })
                    if len(results) >= limit:
                        break
            else:
                diagnostics["duckduckgo_status"] = f"HTTP {resp.status_code}"
        except Exception as exc:
            diagnostics["duckduckgo_error"] = str(exc)
            logger.info("Live web search unavailable (%s); using news fallback", exc)

        # 2. Try Yahoo Finance RSS for Indian equity tickers
        if not results:
            diagnostics["providers_attempted"].append("yahoo_finance_rss")
            ticker_candidates = []
            if required_entities:
                for ent in required_entities:
                    ent_str = str(ent).strip().upper()
                    if ent_str.isalnum() and len(ent_str) <= 12:
                        ticker_candidates.append(f"{ent_str}.NS")
            # Also check words in query
            for word in query.split():
                w = word.strip().upper()
                if w.isalnum() and len(w) <= 10 and w not in ("WHY", "DID", "PROFIT", "FALL", "GROWTH", "REPORT", "Q1", "Q2", "Q3", "Q4"):
                    ticker_candidates.append(f"{w}.NS")

            for ticker in ticker_candidates[:2]:
                try:
                    feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
                    feed_resp = self.session.get(feed_url, timeout=timeout)
                    if feed_resp.status_code == 200:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(feed_resp.content)
                        for it in root.findall(".//item"):
                            title_elem = it.find("title")
                            link_elem = it.find("link")
                            pub_elem = it.find("pubDate")
                            if title_elem is None or not title_elem.text:
                                continue
                            title = clean_text(title_elem.text)
                            link = clean_text(link_elem.text) if link_elem is not None and link_elem.text else ""
                            pub = clean_text(pub_elem.text) if pub_elem is not None and pub_elem.text else None

                            if required_entities and not matches_entity(title, required_entities):
                                continue

                            norm_title = "".join(c.lower() for c in title if c.isalnum())
                            if (link and link in seen_links) or (norm_title and norm_title in seen_titles):
                                continue

                            if link:
                                seen_links.add(link)
                            if norm_title:
                                seen_titles.add(norm_title)

                            domain = urllib.parse.urlparse(link).netloc.replace("www.", "") if link else "Yahoo Finance"
                            results.append({
                                "title": title,
                                "link": link,
                                "source": domain or "Yahoo Finance",
                                "summary": title,
                                "snippet": title,
                                "published": pub,
                                "publication_date": pub,
                                "is_pdf": False,
                            })
                            if len(results) >= limit:
                                break
                    if results:
                        break
                except Exception as yf_exc:
                    diagnostics["yahoo_finance_error"] = str(yf_exc)

        # 3. Try Google News RSS search (reliable public financial news feed)
        if not results:
            diagnostics["providers_attempted"].append("google_news_rss")
            try:
                encoded_q = urllib.parse.quote_plus(query)
                rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
                rss_resp = self.session.get(rss_url, timeout=timeout)
                if rss_resp.status_code == 200:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(rss_resp.content)
                    for item in root.findall(".//item"):
                        title_el = item.find("title")
                        link_el = item.find("link")
                        pub_el = item.find("pubDate")
                        source_el = item.find("source")

                        if title_el is None or not title_el.text:
                            continue
                        title = clean_text(title_el.text)
                        link = clean_text(link_el.text) if link_el is not None and link_el.text else ""
                        pub_date = clean_text(pub_el.text) if pub_el is not None and pub_el.text else None
                        source_name = clean_text(source_el.text) if source_el is not None and source_el.text else "Financial News"

                        combined_text = f"{title} {source_name}"
                        if required_entities and not matches_entity(combined_text, required_entities):
                            continue

                        norm_title = "".join(c.lower() for c in title if c.isalnum())
                        if (link and link in seen_links) or (norm_title and norm_title in seen_titles):
                            continue

                        if link:
                            seen_links.add(link)
                        if norm_title:
                            seen_titles.add(norm_title)

                        results.append({
                            "title": title,
                            "link": link,
                            "source": source_name,
                            "summary": title,
                            "snippet": title,
                            "published": pub_date,
                            "publication_date": pub_date,
                            "is_pdf": False,
                        })
                        if len(results) >= limit:
                            break
            except Exception as gnews_exc:
                diagnostics["google_news_error"] = str(gnews_exc)

        # 4. Fallback to Local News Service if live web search returned nothing
        if not results:
            diagnostics["providers_attempted"].append("local_news_service")
            keywords = [term for term in query.split() if len(term) > 2]
            news_items = self.news_service.search_news(keywords, limit=limit * 2)
            for item in news_items:
                link = item.get("link")
                title = clean_text(item.get("title", ""))
                summary = clean_text(item.get("summary", ""))
                combined_text = f"{title} {summary}"

                if required_entities and not matches_entity(combined_text, required_entities):
                    continue

                norm_title = "".join(c.lower() for c in title if c.isalnum())
                if (link and link in seen_links) or norm_title in seen_titles:
                    continue

                if link:
                    seen_links.add(link)
                if norm_title:
                    seen_titles.add(norm_title)

                results.append({
                    "title": title,
                    "link": link,
                    "source": item.get("source_name") or "News Feed",
                    "summary": summary,
                    "snippet": summary,
                    "published": item.get("published_at"),
                    "publication_date": item.get("published_at"),
                    "is_pdf": False,
                })
                if len(results) >= limit:
                    break

        diagnostics["results_count"] = len(results)
        self.last_search_diagnostics = diagnostics

        self._search_cache[cache_key] = results
        return results

    def fetch_and_extract(
        self,
        url: str,
        title: Optional[str] = None,
        timeout: int = 10,
        max_chars: int = 4000,
    ) -> Optional[Dict[str, Any]]:
        """Fetch accessible HTML page or PDF document and extract clean text and tables."""
        if not url or not url.startswith(("http://", "https://")):
            return None

        if url in self._doc_cache:
            return self._doc_cache[url]

        try:
            resp = self.session.get(url, timeout=timeout, stream=True)
            if resp.status_code != 200:
                logger.debug("URL %s returned status %s", url, resp.status_code)
                return None

            content_type = resp.headers.get("Content-Type", "").lower()
            raw_content = resp.content

            # Check if content is PDF
            is_pdf = (
                "application/pdf" in content_type
                or url.lower().endswith(".pdf")
                or raw_content[:5] == b"%PDF-"
            )

            extracted_text = ""
            doc_title = title or ""
            pub_date = resp.headers.get("Last-Modified")

            if is_pdf and fitz is not None:
                try:
                    pdf_doc = fitz.open(stream=raw_content, filetype="pdf")
                    num_pages = min(len(pdf_doc), 12)
                    pages_text = []
                    for page_idx in range(num_pages):
                        p_text = clean_text(pdf_doc[page_idx].get_text())
                        if p_text:
                            pages_text.append(p_text)
                    extracted_text = "\n\n".join(pages_text)
                    if not doc_title and pdf_doc.metadata.get("title"):
                        doc_title = clean_text(pdf_doc.metadata["title"])
                except Exception as pdf_err:
                    logger.debug("PyMuPDF failed parsing %s: %s", url, pdf_err)
            else:
                # Parse HTML
                try:
                    soup = BeautifulSoup(raw_content, "html.parser")
                    if not doc_title and soup.title:
                        doc_title = clean_text(soup.title.get_text())

                    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
                        tag.decompose()

                    # Convert table tags to readable tabular text representation
                    for table in soup.find_all("table"):
                        rows = []
                        for tr in table.find_all("tr"):
                            cells = [clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
                            if any(cells):
                                rows.append(" | ".join(cells))
                        if rows:
                            table_text = "\n[Table: " + "; ".join(rows[:10]) + "]\n"
                            table.replace_with(table_text)
                        else:
                            table.decompose()

                    # Try to focus on main article container if present
                    main_container = (
                        soup.find("article")
                        or soup.find("main")
                        or soup.find(class_=re.compile(r"article|content|post-body|story", re.I))
                        or soup.body
                        or soup
                    )
                    extracted_text = clean_text(main_container.get_text())
                except Exception as html_err:
                    logger.debug("HTML extraction failed for %s: %s", url, html_err)

            if not extracted_text:
                return None

            # Deduplicate multiple spaces and trim
            truncated_text = extracted_text[:max_chars].strip()

            result = {
                "url": url,
                "title": doc_title or title or "Web Document",
                "content_type": "pdf" if is_pdf else "html",
                "text": truncated_text,
                "publication_date": pub_date,
                "length": len(truncated_text),
                "status": "success",
            }
            self._doc_cache[url] = result
            return result

        except Exception as exc:
            logger.debug("Fetch failed for %s (%s): %s", url, type(exc).__name__, exc)
            return None

    def extract_relevant_excerpts(
        self,
        text: str,
        keywords: List[str],
        max_length: int = 800,
    ) -> str:
        """Find paragraphs or sentences in text that match keywords most strongly."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text

        paragraphs = [clean_text(p) for p in re.split(r"\n\s*\n|\.\s+", text) if len(p.strip()) > 30]
        if not paragraphs:
            return text[:max_length]

        scored: List[Tuple[int, str]] = []
        kw_low = [k.lower() for k in keywords if len(k) > 2]

        for p in paragraphs:
            p_low = p.lower()
            score = sum(2 for k in kw_low if k in p_low)
            # Boost financial driver terms
            if any(term in p_low for term in ("margin", "profit", "revenue", "ebitda", "cost", "expense", "decline", "fell", "grew", "tax", "inflation", "input")):
                score += 3
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        total_len = 0
        for _, p in scored:
            if total_len + len(p) > max_length:
                break
            chosen.append(p)
            total_len += len(p)

        if chosen:
            return " ... ".join(chosen)
        return text[:max_length]


_research_service_instance: Optional[WebResearchService] = None


def get_web_research_service() -> WebResearchService:
    global _research_service_instance
    if _research_service_instance is None:
        _research_service_instance = WebResearchService()
    return _research_service_instance
