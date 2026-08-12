"""
Downloads real WorldPop unconstrained 100m population-count GeoTIFFs
(data.worldpop.org, CC BY 4.0, no registration, verified live before
use) for all 19 project countries.

Verifies each download by actually opening the file with rasterio and
reading the full band -- not just checking byte count, which silently
accepted truncated files in an earlier pass (curl's max-time cutoff was
hit mid-download for 18 of 19 countries, leaving files that looked
complete by size but failed on a real read). No artificial time cap on
curl this time; retries up to 3 times per country if verification fails.
"""
import os
import subprocess
import rasterio

ISO3 = {
    "Afghanistan": "AFG", "Myanmar": "MMR", "Pakistan": "PAK", "Tajikistan": "TJK",
    "Kyrgyzstan": "KGZ", "Uzbekistan": "UZB",
    "Sudan": "SDN", "Ethiopia": "ETH", "Somalia": "SOM", "South Sudan": "SSD",
    "Kenya": "KEN", "Eritrea": "ERI",
    "Colombia": "COL", "Venezuela": "VEN", "Ecuador": "ECU", "Peru": "PER", "Bolivia": "BOL",
    "Haiti": "HTI", "Nicaragua": "NIC",
}
URL_TMPL = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/{iso3}/{iso3_lower}_ppp_2020.tif"
OUT_DIR = "../data_raw/worldpop"


def verify(path):
    try:
        src = rasterio.open(path)
        src.read(1)
        return True
    except Exception:
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for country, iso3 in ISO3.items():
        out_path = os.path.join(OUT_DIR, f"{iso3}_ppp_2020.tif")
        if os.path.exists(out_path) and verify(out_path):
            print(f"  {country} ({iso3}): already downloaded and verified, skipping", flush=True)
            continue

        url = URL_TMPL.format(iso3=iso3, iso3_lower=iso3.lower())
        ok = False
        for attempt in range(1, 6):
            try:
                # Try resume (-C -) first if a partial file exists; but rc=33
                # ("HTTP range error") means this server rejected the resume
                # request -- observed for real on this run -- so on that specific
                # failure, drop the partial file and fall back to a clean,
                # non-resumed download rather than retrying the same broken resume.
                use_resume = os.path.exists(out_path) and os.path.getsize(out_path) > 0
                cmd = ["curl", "-sL", "--fail", "--retry", "3", "--retry-delay", "5", url, "-o", out_path]
                if use_resume:
                    cmd[3:3] = ["-C", "-"]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 33 and use_resume:
                    print(f"  {country} ({iso3}): attempt {attempt} resume rejected (rc=33), "
                          f"dropping partial file and retrying fresh", flush=True)
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    continue
                if result.returncode != 0:
                    print(f"  {country} ({iso3}): attempt {attempt} curl failed (rc={result.returncode})", flush=True)
                    continue
                if verify(out_path):
                    size = os.path.getsize(out_path)
                    print(f"  {country} ({iso3}): {size/1e6:.1f} MB, verified readable", flush=True)
                    ok = True
                    break
                else:
                    print(f"  {country} ({iso3}): attempt {attempt} downloaded but failed verification, retrying", flush=True)
            except Exception as e:
                print(f"  {country} ({iso3}): attempt {attempt} exception ({e})", flush=True)
        if not ok:
            print(f"  {country} ({iso3}): FAILED after 5 attempts", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
