import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

SERIES_HOST_URL = "http://localhost:3000/api/series"
API_KEY = "0fe113aec481fc9f5ba1ffe7a7afe49f"
SCRAPER_URL = "https://api.scraperapi.com/"

# Get series IDs
index_series_request = requests.get(SERIES_HOST_URL)
if index_series_request.status_code == 200:
    series_dict = index_series_request.json()
else:
    raise Exception(f"Failed to fetch series: {index_series_request.status_code}")

series_ids = list(series_dict.keys())
print("series id", series_ids)

#Function to get time series data
def get_timeseries(series_id):
    payload = {
        "seriesid": [series_id],
        "startyear": "2015",
        "endyear": "2024",
    }
    # Send the request via your scraper
    params = { 'api_key': API_KEY, 'url': 'https://api.bls.gov/publicAPI/v2/timeseries/data' }
    
    try:
        response = requests.post(SCRAPER_URL, params=params, json=payload)
        
        if response.status_code == 200:
            # Check if response content is empty
            if not response.text.strip():
                print(f"Empty response for series {series_id}")
                return pd.DataFrame(), False
            
            # Try to parse JSON
            try:
                data = response.json()
            except ValueError as e:
                print(f"Invalid JSON for series {series_id}. Response: {response.text[:200]}...")
                return pd.DataFrame(), False
            
            # Check if the API returned an error in the JSON
            if "Results" not in data:
                print(f"No 'Results' in response for series {series_id}. Response: {data}")
                return pd.DataFrame(), False
            
            # Flatten data into a DataFrame
            rows = []
            for series in data.get("Results", {}).get("series", []):
                for item in series.get("data", []):
                    try:
                        rows.append({
                            "series_id": series_id,
                            "year": int(item["year"]),
                            "period": item["period"],
                            "period_name": item["periodName"],
                            "value": float(item["value"])
                        })
                    except (KeyError, ValueError) as e:
                        print(f"Error parsing data item for series {series_id}: {e}")
                        continue
            
            return pd.DataFrame(rows), True  # Return success flag
        else:
            print(f"HTTP error for series {series_id}: {response.status_code} - {response.text[:200]}")
            return pd.DataFrame(), False  # Return failure flag
            
    except requests.exceptions.RequestException as e:
        print(f"Request exception for series {series_id}: {e}")
        return pd.DataFrame(), False

dfs = []
failed_series = []
request_count = 0

# Set the number of parallel threads (adjust based on API rate limits)
MAX_WORKERS = 5

print(f"Starting parallel processing with {MAX_WORKERS} workers...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all requests
    future_to_series = {executor.submit(get_timeseries, sid): sid for sid in series_ids}
    
    # Process completed requests as they finish
    for future in as_completed(future_to_series):
        series_id = future_to_series[future]
        request_count += 1
        
        try:
            df, success = future.result()
            
            if success and not df.empty:
                print(f"Request #{request_count} SUCCEEDED - Series ID: {series_id}")
                dfs.append(df)
            else:
                print(f"Request #{request_count} FAILED - Series ID: {series_id}")
                failed_series.append(series_id)
                
        except Exception as exc:
            print(f"Request #{request_count} FAILED - Series ID: {series_id} - Exception: {exc}")
            failed_series.append(series_id)

if dfs:
    full_df = pd.concat(dfs, ignore_index=True)
    print(full_df.head())
    
    # Export as JSON
    full_json = full_df.to_json(orient="records", date_format="iso")
    with open("full_timeseries.json", "w") as f:
        f.write(full_json)
    print("JSON file saved as full_timeseries.json")
else:
    print("No data fetched.")

# Display all failed series identifiers
print("\n" + "="*50)
print("SUMMARY:")
print(f"Total requests: {request_count}")
print(f"Successful requests: {len(dfs)}")
print(f"Failed requests: {len(failed_series)}")

if failed_series:
    print(f"\nFailed Series Identifiers:")
    for failed_id in failed_series:
        print(f"  - {failed_id}")
else:
    print("\nAll requests succeeded!")
print("="*50)