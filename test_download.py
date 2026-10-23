
import requests
from pathlib import Path

url = (
    "https://nsearchives.nseindia.com/corporate/"
    "Infosys_29052026201312_"
    "Infosys_Integrated_Annual_Report_2025-26.pdf"
)

output = Path("finaudit_test/infosys_fy2025_26.pdf")
output.parent.mkdir(parents=True, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.nseindia.com/",
}

try:
    response = requests.get(
        url,
        headers=headers,
        timeout=60
    )

    print("HTTP status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Bytes received:", len(response.content))

    if response.status_code == 200 and response.content.startswith(b"%PDF-"):
        output.write_bytes(response.content)
        print("SUCCESS: PDF downloaded")
        print("Location:", output.resolve())
        print("Size:", f"{output.stat().st_size:,} bytes")
    else:
        print("FAILED: Response was not a valid PDF.")
        print(response.text[:300] if "text" in response.headers.get(
            "Content-Type", ""
        ).lower() else "No HTML error body.")

except requests.RequestException as exc:
    print("Request failed:", repr(exc))
