
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Official NSE annual reports page
PAGE_URL = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-filings-annual-reports"
)

COMPANY = "infosys"
SAVE_DIR = "fin_audit_test_reports"
os.makedirs(SAVE_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,*/*",
    "Referer": "https://www.nseindia.com/"
})

def test_nse_download():
    print("1. Opening NSE website...")

    try:
        # Load homepage first to establish session cookies
        home = session.get(
            "https://www.nseindia.com/",
            timeout=30
        )
        print("Homepage status:", home.status_code)

        response = session.get(PAGE_URL, timeout=30)
        print("Annual reports page status:", response.status_code)
        response.raise_for_status()

    except requests.RequestException as exc:
        print("FAILED to access NSE:", exc)
        return

    soup = BeautifulSoup(response.text, "html.parser")
    candidates = []

    # Inspect links rendered in the returned HTML
    for a in soup.find_all("a", href=True):
        href = urljoin(PAGE_URL, a["href"])
        text = a.get_text(" ", strip=True)
        combined = f"{text} {href}".lower()

        if COMPANY in combined and (
            ".pdf" in urlparse(href).path.lower()
            or "annual" in combined
            or "report" in combined
            or "download" in combined
        ):
            candidates.append((text, href))

    # Remove duplicate URLs
    candidates = list(dict.fromkeys(candidates))
    print("Matching report links found:", len(candidates))

    if not candidates:
        print(
            "\nNo matching links were found in the HTML response."
            "\nThe website may load filings dynamically using JavaScript "
            "or a separate API. This does NOT prove the reports are "
            "unavailable."
        )
        with open("nse_response.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Saved response HTML to nse_response.html for inspection.")
        return

    successful = 0

    for index, (label, url) in enumerate(candidates, start=1):
        print(f"\nTesting link {index}: {label or '(no link text)'}")
        print(url)

        try:
            file_response = session.get(
                url,
                timeout=60,
                stream=True
            )
            content_type = file_response.headers.get(
                "Content-Type", ""
            ).lower()

            # Verify the downloaded content is actually a PDF
            first_chunk = next(
                file_response.iter_content(chunk_size=8192),
                b""
            )

            is_pdf = (
                first_chunk.startswith(b"%PDF-")
                or "application/pdf" in content_type
            )

            if file_response.status_code != 200 or not is_pdf:
                print(
                    "Not a confirmed PDF:",
                    file_response.status_code,
                    content_type
                )
                file_response.close()
                continue

            filename = f"infosys_report_{index}.pdf"
            path = os.path.join(SAVE_DIR, filename)

            with open(path, "wb") as f:
                f.write(first_chunk)
                for chunk in file_response.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            file_response.close()
            size = os.path.getsize(path)

            if size > 1000:
                print("SUCCESS:", path, f"({size:,} bytes)")
                successful += 1
            else:
                os.remove(path)
                print("Rejected: file is unexpectedly small.")

        except requests.RequestException as exc:
            print("Download failed:", exc)

    print("\n========== TEST RESULT ==========")
    print("Matching links:", len(candidates))
    print("Confirmed PDFs downloaded:", successful)
    print("Output directory:", os.path.abspath(SAVE_DIR))

    if successful:
        print("PASS: Automated PDF downloading worked.")
    else:
        print(
            "INCONCLUSIVE: No PDFs were downloaded from the "
            "links found. Inspect the saved HTML or use the "
            "official filing data endpoint if available."
        )

test_nse_download()
