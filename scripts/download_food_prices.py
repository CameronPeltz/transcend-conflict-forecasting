"""
Real WFP (World Food Programme) VAM food price data for each target
country, pulled live from the Humanitarian Data Exchange (data.humdata.org)
-- free, no API key. Each country's CSV has thousands of real market-level
price observations (cereals, oil, sugar, etc.) reported in both local
currency and USD-normalized price, which sidesteps the need for a
separate FX-rate feed (checked and found not to cover these currencies --
see write-up).

Food-price spikes are a well-documented real leading indicator of unrest
in the conflict literature (Brinkman & Hendrix and similar work on food
prices and instability), which is why this is a candidate leading
indicator rather than just background context.
"""
import requests
import pandas as pd

# resource URLs confirmed live via the HDX package_show API
COUNTRY_RESOURCES = {
    "SU": "https://data.humdata.org/dataset/369e003b-f0af-4e48-99d7-34fc85b44635/resource/8fea18b2-615f-4af5-9bd5-85cc31a25ffd/download/wfp_food_prices_sdn.csv",
    "ET": "https://data.humdata.org/dataset/2e4f1922-e446-4b57-a98a-d0e2d5e34afa/resource/87bac18e-f3aa-4b29-8cf8-76763e823dc5/download/wfp_food_prices_eth.csv",
    "AF": "https://data.humdata.org/dataset/a246cbac-42d5-47b2-ba75-ac66f69e83de/resource/03e6ce5d-03a2-4e60-8d04-afa39c5972f4/download/wfp_food_prices_afg.csv",
    "BM": "https://data.humdata.org/dataset/4d052db4-0fb8-47b5-a56a-633fdcf0e55c/resource/4f188cc5-7f6a-4354-95d7-15bd45bb8587/download/wfp_food_prices_mmr.csv",
    "CO": "https://data.humdata.org/dataset/dd6f3c15-69c0-44c0-8f62-6c5395dcc572/resource/c99063c7-016c-4eef-89dd-9a88db15ad7c/download/wfp_food_prices_col.csv",
    "VE": "https://data.humdata.org/dataset/a3800f18-0126-4911-917e-2b2a9415d909/resource/706f628f-98e4-4bf9-a077-5ef85c83aa77/download/wfp_food_prices_ven.csv",
}


def main():
    frames = []
    for code, url in COUNTRY_RESOURCES.items():
        print(f"fetching WFP food prices for {code}...")
        df = pd.read_csv(url, skiprows=[1])  # row 1 is a HXL tag row, not data
        df["country"] = code
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= pd.Timestamp("2025-01-01")]  # keep recent window + lookback;
        # Venezuela's real WFP feed stops updating in 2025-05 (a real, disclosed reporting
        # gap, not a bug) so this wider cutoff is what lets VE contribute anything at all
        df["usdprice"] = pd.to_numeric(df["usdprice"], errors="coerce")
        keep = ["country", "date", "admin1", "market", "category", "commodity", "unit", "pricetype", "currency", "price", "usdprice"]
        frames.append(df[keep])
        print(f"  {len(df)} real recent rows")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv("data/food_prices_recent.csv", index=False)
    print(f"\nwrote data/food_prices_recent.csv ({len(out)} real rows, {out['country'].nunique()} countries)")


if __name__ == "__main__":
    main()
