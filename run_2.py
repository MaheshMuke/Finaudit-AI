
import asyncio
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright

URL = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-filings-annual-reports"
    "?symbol=INFY&tabIndex=equity"
)

OUT = Path("finaudit_download_test")
OUT.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            accept_downloads=True
        )
        page = await context.new_page()

        print("Opening official NSE annual-report page...")

        try:
            response = await page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=60000
            )
            print("HTTP status:", response.status if response else "Unknown")
            await page.wait_for_timeout(8000)

        except Exception as e:
            print("Page loading failed:", e)
            await browser.close()
            return

        # Save rendered page for debugging
        (OUT / "rendered_page.html").write_text(
            await page.content(),
            encoding="utf-8"
        )

        # Inspect the rendered report table
        rows = await page.locator("tr").all()
        print("\nVisible report rows:")

        for row in rows[:12]:
            text = " ".join((await row.inner_text()).split())
            if text:
                print("-", text[:250])

        # Collect direct links and their visible labels
        links = await page.locator("a[href]").evaluate_all("""
            els => els.map(a => ({
                text: (a.innerText || a.getAttribute('aria-label') || '').trim(),
                href: a.href
            }))
        """)

        candidates = []
        for item in links:
            href = item["href"]
            label = item["text"]
            combined = (label + " " + href).lower()

            if (
                ".pdf" in urlparse(href).path.lower()
                or "download" in combined
                or "attachment" in combined
                or re.search(r"\\d+(\\.\\d+)?\\s*mb", combined)
            ):
                candidates.append(item)

        # Remove duplicate URLs
        unique = {}
        for item in candidates:
            unique[item["href"]] = item
        candidates = list(unique.values())

        print("\nPotential report links found:", len(candidates))

        for item in candidates[:20]:
            print(item["text"][:80], "->", item["href"])

        downloaded = 0

        # Try accessible direct links, using browser-session cookies
        for i, item in enumerate(candidates, start=1):
            href = item["href"]

            try:
                result = await context.request.get(
                    href,
                    timeout=60000
                )

                if not result.ok:
                    print("HTTP failure:", result.status, href)
                    continue

                body = await result.body()
                content_type = result.headers.get(
                    "content-type", ""
                ).lower()

                # Confirm the file is a PDF, not an HTML error page
                if not body.startswith(b"%PDF-"):
                    print(
                        "Not a PDF:",
                        content_type,
                        href
                    )
                    continue

                path = OUT / f"report_{i}.pdf"
                path.write_bytes(body)

                print(
                    "SUCCESS:",
                    path,
                    f"({len(body):,} bytes)"
                )
                downloaded += 1

            except Exception as e:
                print("Download error:", href, str(e)[:200])

        print("\n========== FINAL RESULT ==========")
        print("Potential links:", len(candidates))
        print("Confirmed PDFs:", downloaded)
        print("Debug HTML:", OUT / "rendered_page.html")

        if downloaded:
            print("PASS: At least one report was downloaded.")
        else:
            print(
                "NO DIRECT PDF FOUND: Inspect rendered_page.html "
                "and the printed links. The attachment may require "
                "a JavaScript action or a different official endpoint."
            )

        await browser.close()


asyncio.run(main())
