
import re
import pdfplumber
from pathlib import Path

PDF_PATH = Path("finaudit_test/infosys_fy2025_26.pdf")
OUTPUT_PATH = Path("finaudit_output/statement_inspection.txt")

PATTERNS = {
    "Income Statement": [
        r"consolidated statement of profit and loss",
        r"consolidated statement of comprehensive income",
    ],
    "Balance Sheet": [
        r"consolidated balance sheet",
        r"consolidated statement of financial position",
    ],
    "Cash Flow": [
        r"consolidated statement of cash flows",
        r"consolidated cash flow statement",
    ],
}

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def main():
    if not PDF_PATH.exists():
        print("PDF not found:", PDF_PATH.resolve())
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches_found = 0

    with pdfplumber.open(PDF_PATH) as pdf, \
            OUTPUT_PATH.open("w", encoding="utf-8") as out:

        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            normalized = normalize(text)

            matches = []
            for statement, patterns in PATTERNS.items():
                if any(re.search(p, normalized) for p in patterns):
                    matches.append(statement)

            if not matches:
                continue

            matches_found += 1

            header = (
                f"\n{'=' * 70}\n"
                f"PDF PAGE: {index}\n"
                f"MATCHES: {', '.join(matches)}\n"
                f"{'=' * 70}\n"
            )

            print(header.strip())
            out.write(header)
            out.write("\nPAGE TEXT:\n")
            out.write(text[:5000])
            out.write("\n\nEXTRACTED TABLES:\n")

            tables = page.extract_tables()

            if not tables:
                out.write("No tables detected by pdfplumber.\n")

            for table_number, table in enumerate(tables, start=1):
                out.write(f"\nTable {table_number}:\n")
                for row in table:
                    out.write(
                        " | ".join(
                            str(cell or "").replace("\n", " ")
                            for cell in row
                        ) + "\n"
                    )

    print("\nCandidate pages found:", matches_found)
    print("Inspection file:", OUTPUT_PATH.resolve())

if __name__ == "__main__":
    main()

