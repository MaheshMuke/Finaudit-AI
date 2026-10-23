
import pdfplumber
from pathlib import Path

PDF_PATH = Path("finaudit_test/infosys_fy2025_26.pdf")

# Python uses zero-based indexes internally, but we display
# the human-readable PDF page numbers.
PAGES_TO_INSPECT = [80, 276, 277, 281, 286, 288, 295, 296]

with pdfplumber.open(PDF_PATH) as pdf:
    for page_number in PAGES_TO_INSPECT:
        page = pdf.pages[page_number - 1]

        print("\n" + "=" * 80)
        print(f"PDF PAGE {page_number}")
        print("=" * 80)

        text = page.extract_text() or ""
        print("\n--- EXTRACTED TEXT ---")
        print(text[:3500] if text else "No text extracted")

        tables = page.extract_tables()
        print(f"\n--- TABLES FOUND: {len(tables)} ---")

        for i, table in enumerate(tables, start=1):
            print(f"\nTable {i}:")
            for row in table[:35]:
                print(" | ".join(str(cell or "") for cell in row))
