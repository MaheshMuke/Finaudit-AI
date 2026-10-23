
import re
import json
from pathlib import Path

import pdfplumber

PDF_PATH = Path(
    "finaudit_test/infosys_fy2025_26.pdf"
)
OUTPUT_DIR = Path("finaudit_output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keywords that help locate financial statements
STATEMENT_PATTERNS = {
    "income_statement": [
        r"statement of profit and loss",
        r"statement of comprehensive income",
        r"consolidated statement of profit",
    ],
    "balance_sheet": [
        r"balance sheet",
        r"statement of financial position",
    ],
    "cash_flow": [
        r"statement of cash flows",
        r"cash flow statement",
        r"cash flows from operating activities",
    ],
}

def normalize(text):
    """Normalize whitespace for reliable keyword matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def identify_statements(text):
    normalized = normalize(text)
    found = []

    for statement, patterns in STATEMENT_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            found.append(statement)

    return found


def main():
    if not PDF_PATH.exists():
        print(f"PDF NOT FOUND: {PDF_PATH.resolve()}")
        print("Check the PDF path before running again.")
        return

    all_pages = []
    statement_pages = {
        name: [] for name in STATEMENT_PATTERNS
    }

    print("Opening:", PDF_PATH.resolve())

    with pdfplumber.open(PDF_PATH) as pdf:
        print("Total PDF pages:", len(pdf.pages))

        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            all_pages.append({
                "page": page_number,
                "text": text
            })

            matches = identify_statements(text)

            for statement in matches:
                statement_pages[statement].append(page_number)

            if matches:
                print(
                    f"Page {page_number}: "
                    f"{', '.join(matches)}"
                )

    # Save all extracted text for debugging and later processing.
    text_path = OUTPUT_DIR / "extracted_pages.txt"

    with text_path.open("w", encoding="utf-8") as file:
        for item in all_pages:
            file.write(
                f"\n{'=' * 25} "
                f"PDF PAGE {item['page']} "
                f"{'=' * 25}\n"
            )
            file.write(item["text"])
            file.write("\n")

    # Save a machine-readable index for future agent tools.
    index_path = OUTPUT_DIR / "statement_index.json"

    with index_path.open("w", encoding="utf-8") as file:
        json.dump(
            statement_pages,
            file,
            indent=2
        )

    print("\n========== EXTRACTION SUMMARY ==========")

    for statement, pages in statement_pages.items():
        print(f"{statement}: {len(pages)} candidate pages")
        if pages:
            print("  Pages:", pages)

    print("\nExtracted text:", text_path.resolve())
    print("Statement index:", index_path.resolve())

    if not any(statement_pages.values()):
        print(
            "\nNo statement headings detected. "
            "The PDF may use different wording or scanned pages."
        )


if __name__ == "__main__":
    main()
