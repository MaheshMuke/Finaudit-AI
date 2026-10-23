export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface CompanyRecord {
  symbol: string;
  company_name: string;
  industry: string;
  series?: string;
  isin_code?: string;
}

export interface CommodityItem {
  name: string;
  symbol: string;
  unit: string;
  category: string;
  last_price: number | null;
  change_pct: number | null;
  status: string;
  retrieved_at?: string;
}

export interface NewsArticle {
  article_id?: string;
  id?: number | string;
  source_name: string;
  category: string;
  title: string;
  link: string;
  summary: string;
  published_at: string;
}

export interface WatchlistItem {
  symbol: string;
  company_name: string;
  list_name: string;
  quantity?: number | null;
  average_buy_price?: number | null;
  created_at: string;
}

export interface CompanyImpact {
  symbol: string;
  company_name: string;
  impact: string;
  materiality: string;
  mechanism: string;
}

export interface ImpactAnalysis {
  short_answer: string;
  macro_driver?: string;
  transmission_mechanism: string;
  company_impacts: CompanyImpact[];
  indicators_to_monitor: string[];
  confidence_and_limitations?: string;
}

export interface PortfolioImpactResponse {
  success: boolean;
  analysis: ImpactAnalysis;
  news_evidence: NewsArticle[];
  financial_evidence?: Array<Record<string, string | number | null>>;
}

export interface SectorRadarItem {
  industry: string;
  opportunity_score: number;
  median_revenue_growth: number;
  median_net_margin: number | null;
  median_roe: number | null;
  median_pe?: number | null;
  company_count: number;
  sample_size?: number;
  reporting_periods?: string[];
  methodology?: string;
}

export interface AnalysisResult {
  success: boolean;
  company: {
    symbol: string;
    company_name: string;
    industry: string;
    currency: string;
    source: string;
  };
  market_data: {
    current_price: number | null;
    previous_close?: number | null;
    currency?: string;
    source?: string;
    retrieved_at?: string;
    status?: string;
    one_year_return: number | null;
    annualized_volatility: number | null;
    max_drawdown: number | null;
    fifty_two_week_high: number | null;
    fifty_two_week_low: number | null;
    history_dates: string[];
    history_prices: number[];
  };
  calculations: {
    latest_ratios: Record<string, string | number | null>;
    metric_metadata?: {
      source?: string;
      original_source?: string | null;
      cache_layer?: string;
      source_warning?: string;
      currency?: string;
      raw_unit?: string;
      display_unit?: string;
      reporting_period?: string | null;
      previous_period?: string | null;
      statement_basis?: string;
      per_metric?: Record<string, {
        reporting_period?: string | null;
        previous_period?: string | null;
        statement_basis?: string;
        currency?: string;
        unit?: string;
      }>;
    };
    annual_series: {
      periods: string[];
      revenue: (number | null)[];
      net_income: (number | null)[];
      operating_income: (number | null)[];
      free_cash_flow: (number | null)[];
    };
    quarterly_series?: {
      periods: string[];
      revenue: (number | null)[];
      net_income: (number | null)[];
      operating_income: (number | null)[];
      free_cash_flow: (number | null)[];
    };
  };
  competitors: {
    peers: CompanyRecord[];
    comparison_table: Array<Record<string, string | number | boolean | null>>;
    rankings: Record<string, { rank: number; out_of: number; leader: string }>;
    comparison_metadata?: {
      basis?: string;
      target_period?: string;
      reporting_basis?: string;
      reporting_periods?: string[];
      note?: string;
      peer_selection_method?: string;
      peer_selection_confidence?: string;
    };
  };
  sector_framework: {
    sector: string;
    key_metrics: string[];
    irrelevant_or_low_relevance: string[];
    guidance: string;
  };
  scenarios: Record<string, unknown>;
  ai_intelligence: {
    executive_summary?: string;
    business_model?: {
      summary?: string;
      revenue_generation?: string;
    };
    business_dependencies?: {
      customers_and_concentration?: string;
      suppliers_and_input_costs?: string;
      macro_commodities_fx_rates?: string;
    };
    share_price_drivers?: {
      positive_catalysts?: string[];
      negative_risks?: string[];
    };
    strengths?: string[];
    weaknesses?: string[];
    limitations?: string[];
  };
  data_metadata?: {
    financial?: AnalysisResult["calculations"]["metric_metadata"];
    market?: { source?: string; retrieved_at?: string; status?: string };
  };
  evidence_validation?: {
    status: string;
    disclosures: string[];
    checked_items_count: number;
  };
  ai_error?: string | null;
  llm_provider?: string | null;
  llm_metadata?: {
    request_id?: string;
    provider?: string | null;
    model?: string | null;
    generation_status?: string;
    fallback_used?: boolean;
    latency_ms?: number;
    error_code?: string;
  } | null;
}
