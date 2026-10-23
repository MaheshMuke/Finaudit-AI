from typing import Any, Dict, List, Optional
from finaudit.financial_calc import format_currency_inr, format_pct


class SectorIntelligence:
    """
    Sector-specific metrics, weights, and analytical framework definitions.
    Customizes focus between Banking/NBFCs, IT Services, Manufacturing, Energy, etc.
    """

    SECTOR_FRAMEWORKS = {
        "Financial Services": {
            "key_metrics": ["ROE", "ROA", "Net Interest Margin / Spread", "Loan Book Growth", "Credit Quality / NPA"],
            "not_applicable": ["Gross Margin", "Current Ratio", "Inventory Turnover", "EBITDA"],
            "primary_driver": "Interest rate cycles, credit demand, deposit franchise, and provisioning costs.",
        },
        "Information Technology": {
            "key_metrics": ["Revenue Growth (YoY)", "Operating Margin (EBIT)", "Free Cash Flow Conversion", "Attrition / Utilisation", "USD/INR Sensitivity"],
            "not_applicable": ["Inventory Turnover", "Debt to Equity (usually negligible)"],
            "primary_driver": "Global enterprise tech spend, digital transformation budgets, and currency tailwinds.",
        },
        "Oil Gas & Consumable Fuels": {
            "key_metrics": ["Operating Cash Flow", "EBITDA Margin", "CapEx Intensity", "Refining / Cracking Margins", "Net Debt to EBITDA"],
            "not_applicable": ["R&D to Revenue", "Customer Acquisition Cost"],
            "primary_driver": "Crude benchmarks (Brent/WTI), crack spreads, government windfall taxes, and downstream demand.",
        },
        "Automobile and Auto Components": {
            "key_metrics": ["Volume Growth", "Gross Margin", "Operating Margin", "Working Capital Days", "CapEx / EV Transition"],
            "not_applicable": ["Loan Loss Provisioning", "Net Interest Margin"],
            "primary_driver": "Consumer sentiment, raw-material inflation (steel, aluminium), semiconductor supply, and EV adoption.",
        },
        "Healthcare": {
            "key_metrics": ["Operating Margin", "R&D Spend (% of Rev)", "US Generic Price Erosion", "Domestic Formulation Growth", "ROIC"],
            "not_applicable": ["Commodity Price Pass-Through", "Inventory Liquidity Ratio"],
            "primary_driver": "Regulatory inspections (USFDA), patent expirations, ANDA approvals, and chronic healthcare demand.",
        },
    }

    @classmethod
    def get_framework(cls, industry: str) -> Dict[str, Any]:
        for sec, data in cls.SECTOR_FRAMEWORKS.items():
            if sec.lower() in industry.lower() or industry.lower() in sec.lower():
                return data

        # Default industrial framework
        return {
            "key_metrics": ["Revenue Growth", "Operating Margin", "Debt to Equity", "Free Cash Flow", "Return on Equity (ROE)"],
            "not_applicable": [],
            "primary_driver": "Macroeconomic demand, competitive pricing power, and input cost trends.",
        }
