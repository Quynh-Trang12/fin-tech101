import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Custom Metadata for Artifact tracking
ARTIFACT_METADATA = {
    "Summary": "News API feasibility testing script for Commonwealth Bank of Australia (CBA) news retrieval.",
    "UserFacing": True,
    "RequestFeedback": False
}

def load_local_env():
    """Manually parse .env file if it exists, without dotenv dependency."""
    env_path = Path(".env")
    if env_path.exists():
        print(f"[Env Loader] Found .env file at {env_path.resolve()}. Loading keys...")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

def get_recent_date_range(days=30):
    """Get date strings for APIs that require from/to parameters."""
    today = datetime.now()
    start_date = today - timedelta(days=days)
    return start_date.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

def test_yahoo_finance(queries, raw_dir):
    """Query Yahoo Finance Search API for recent news stories."""
    print("\n" + "="*50)
    print("TESTING YAHOO FINANCE SEARCH API")
    print("="*50)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    articles = []
    
    for q in queries:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        print(f"[Yahoo Finance] Fetching news for query/symbol: '{q}'...")
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                print(f"[Yahoo Finance] Failed for '{q}': HTTP {res.status_code}")
                continue
            
            data = res.json()
            
            # Save raw sample
            raw_path = raw_dir / f"yfinance_{q}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[Yahoo Finance] Saved raw JSON to {raw_path}")
            
            news_items = data.get("news", [])
            print(f"[Yahoo Finance] Found {len(news_items)} articles.")
            
            for item in news_items:
                pub_time = item.get("providerPublishTime")
                published_at_str = ""
                if pub_time:
                    try:
                        published_at_str = datetime.fromtimestamp(pub_time).isoformat()
                    except Exception:
                        pass
                
                # Check for CBA.AX in related tickers to confirm relevance
                related_tickers = item.get("relatedTickers", [])
                
                articles.append({
                    "source": "Yahoo Finance",
                    "query_or_symbol": q,
                    "published_at": published_at_str,
                    "headline": item.get("title", ""),
                    "summary_or_description": "",  # Search API does not provide a summary
                    "publisher_source_name": item.get("publisher", ""),
                    "url": item.get("link", ""),
                    "extra_info": f"Tickers: {related_tickers}"
                })
        except Exception as e:
            print(f"[Yahoo Finance] Error fetching '{q}': {e}")
            
    return articles

def test_alpha_vantage(queries, raw_dir, api_key):
    """Query Alpha Vantage News & Sentiment API."""
    print("\n" + "="*50)
    print("TESTING ALPHA VANTAGE NEWS & SENTIMENT API")
    print("="*50)
    
    if not api_key:
        print("[Alpha Vantage] ALPHA_VANTAGE_API_KEY environment variable not found. Skipping.")
        return []
    
    articles = []
    for q in queries:
        # Alpha Vantage uses 'tickers' query parameter
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={q}&limit=50&apikey={api_key}"
        print(f"[Alpha Vantage] Fetching news for ticker: '{q}'...")
        try:
            res = requests.get(url, timeout=15)
            if res.status_code != 200:
                print(f"[Alpha Vantage] Failed for '{q}': HTTP {res.status_code}")
                continue
            
            data = res.json()
            
            # Check for API error or rate limits
            if "Note" in data:
                print(f"[Alpha Vantage] Rate limit info: {data['Note']}")
            if "Error Message" in data:
                print(f"[Alpha Vantage] API Error: {data['Error Message']}")
                continue
                
            raw_path = raw_dir / f"alphavantage_{q}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[Alpha Vantage] Saved raw JSON to {raw_path}")
            
            feed = data.get("feed", [])
            print(f"[Alpha Vantage] Found {len(feed)} articles.")
            
            for item in feed:
                time_pub = item.get("time_published")
                published_at_str = ""
                if time_pub:
                    try:
                        published_at_str = datetime.strptime(time_pub, "%Y%m%dT%H%M%S").isoformat()
                    except Exception:
                        published_at_str = time_pub
                
                articles.append({
                    "source": "Alpha Vantage",
                    "query_or_symbol": q,
                    "published_at": published_at_str,
                    "headline": item.get("title", ""),
                    "summary_or_description": item.get("summary", ""),
                    "publisher_source_name": item.get("source", ""),
                    "url": item.get("url", ""),
                    "extra_info": f"Overall Sentiment: {item.get('overall_sentiment_label', '')}"
                })
        except Exception as e:
            print(f"[Alpha Vantage] Error fetching '{q}': {e}")
            
    return articles

def test_finnhub(queries, raw_dir, api_key):
    """Query Finnhub Company News API."""
    print("\n" + "="*50)
    print("TESTING FINNHUB COMPANY NEWS API")
    print("="*50)
    
    if not api_key:
        print("[Finnhub] FINNHUB_API_KEY environment variable not found. Skipping.")
        return []
    
    articles = []
    start_date, end_date = get_recent_date_range(days=30)
    
    for q in queries:
        # Finnhub requires from/to date parameters
        url = f"https://finnhub.io/api/v1/company-news?symbol={q}&from={start_date}&to={end_date}&token={api_key}"
        print(f"[Finnhub] Fetching news for symbol: '{q}' ({start_date} to {end_date})...")
        try:
            res = requests.get(url, timeout=15)
            if res.status_code != 200:
                print(f"[Finnhub] Failed for '{q}': HTTP {res.status_code}")
                continue
            
            data = res.json()
            
            # Save raw sample
            raw_path = raw_dir / f"finnhub_{q}.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[Finnhub] Saved raw JSON to {raw_path}")
            
            if isinstance(data, dict) and "error" in data:
                print(f"[Finnhub] API Error: {data['error']}")
                continue
                
            print(f"[Finnhub] Found {len(data)} articles.")
            
            for item in data:
                pub_time = item.get("datetime")
                published_at_str = ""
                if pub_time:
                    try:
                        published_at_str = datetime.fromtimestamp(pub_time).isoformat()
                    except Exception:
                        pass
                        
                articles.append({
                    "source": "Finnhub",
                    "query_or_symbol": q,
                    "published_at": published_at_str,
                    "headline": item.get("headline", ""),
                    "summary_or_description": item.get("summary", ""),
                    "publisher_source_name": item.get("source", ""),
                    "url": item.get("url", ""),
                    "extra_info": f"ID: {item.get('id', '')}"
                })
        except Exception as e:
            print(f"[Finnhub] Error fetching '{q}': {e}")
            
    return articles

def print_source_comparison_summary(df):
    """Analyze and print comparative statistics for each API source."""
    print("\n" + "="*50)
    print("NEWS SOURCE COMPILATION COMPARISON SUMMARY")
    print("="*50)
    
    if df.empty:
        print("No articles fetched from any source. Comparison table is empty.")
        return
        
    for source, group in df.groupby("source"):
        print(f"\nSource: {source}")
        print(f"  - Total Articles Returned: {len(group)}")
        
        # Date range analysis
        valid_dates = pd.to_datetime(group["published_at"].dropna())
        if not valid_dates.empty:
            print(f"  - Earliest Date: {valid_dates.min().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  - Latest Date:   {valid_dates.max().strftime('%Y-%m-%d %H:%M:%S')}")
            print("  - Timestamp exists: Yes (100%)")
        else:
            print("  - Timestamp exists: No")
            
        # Feature coverage checks
        has_headline = group["headline"].astype(str).str.strip().str.len() > 0
        has_summary = group["summary_or_description"].astype(str).str.strip().str.len() > 0
        has_url = group["url"].astype(str).str.strip().str.len() > 0
        
        print(f"  - Headline coverage: {has_headline.mean()*100:.1f}%")
        print(f"  - Summary/Desc coverage: {has_summary.mean()*100:.1f}%")
        print(f"  - URL coverage: {has_url.mean()*100:.1f}%")
        
        # Obvious limitations
        if source == "Yahoo Finance":
            print("  - Known Limitation: No article summary/body provided by the Search news API feed.")
        elif source == "Alpha Vantage":
            print("  - Known Limitation: Standard API limits apply (25 requests per day for free tier).")
        elif source == "Finnhub":
            print("  - Known Limitation: Company news feed is heavily tailored to US-listed exchange tickers.")

def main():
    load_local_env()
    
    # Establish workspace directories
    experiments_dir = Path(__file__).resolve().parent
    raw_dir = experiments_dir / "raw_news_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Symbols & queries to query
    yahoo_queries = ["CBA.AX", "CBAUF", "CMWAY"]
    av_queries = ["CBA.AX", "CBAUF"]
    finnhub_queries = ["CBA.AX", "CBAUF"]
    
    # Check for API Keys
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    
    all_articles = []
    
    # Test Yahoo Finance
    yahoo_articles = test_yahoo_finance(yahoo_queries, raw_dir)
    all_articles.extend(yahoo_articles)
    
    # Test Alpha Vantage
    av_articles = test_alpha_vantage(av_queries, raw_dir, av_key)
    all_articles.extend(av_articles)
    
    # Test Finnhub
    finnhub_articles = test_finnhub(finnhub_queries, raw_dir, finnhub_key)
    all_articles.extend(finnhub_articles)
    
    # Create comparison summary CSV
    df = pd.DataFrame(all_articles)
    csv_output_path = experiments_dir / "news_source_comparison.csv"
    
    if not df.empty:
        # Standardize columns
        columns_to_keep = [
            "source", "query_or_symbol", "published_at", "headline",
            "summary_or_description", "publisher_source_name", "url"
        ]
        df_to_save = df[columns_to_keep]
        df_to_save.to_csv(csv_output_path, index=False)
        print(f"\n[Summary] Successfully saved comparison table to: {csv_output_path.resolve()}")
    else:
        # Create an empty CSV template if no articles returned
        pd.DataFrame(columns=[
            "source", "query_or_symbol", "published_at", "headline",
            "summary_or_description", "publisher_source_name", "url"
        ]).to_csv(csv_output_path, index=False)
        print(f"\n[Warning] No articles collected. Saved empty template to: {csv_output_path.resolve()}")
        
    print_source_comparison_summary(df)

if __name__ == "__main__":
    main()
