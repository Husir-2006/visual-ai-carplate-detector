import argparse
import urllib.request
from pathlib import Path


ZENODO_RECORD = "15647076"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def fetch_record():
    try:
        import requests
    except Exception:
        requests = None
    if requests:
        response = requests.get(ZENODO_API, timeout=30)
        response.raise_for_status()
        return response.json()
    import json
    with urllib.request.urlopen(ZENODO_API, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url, output, expected_size=0):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and expected_size and output.stat().st_size >= expected_size * 0.995:
        print(f"skip existing: {output}")
        return
    if output.exists():
        print(f"remove incomplete file: {output}")
        output.unlink()

    try:
        import requests
    except Exception:
        requests = None

    if requests:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", expected_size or 0))
            done = 0
            with output.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\rdownloading {output.name}: {done * 100 / total:5.1f}%", end="")
        print(f"\ndownloaded: {output}")
        return

    urllib.request.urlretrieve(url, output)
    print(f"downloaded: {output}")


def main():
    parser = argparse.ArgumentParser(description="Download CCPD Chinese license plate dataset archive from Zenodo.")
    parser.add_argument("--target", default="datasets/CCPD", help="Output directory.")
    parser.add_argument("--contains", default="CCPD2019", help="Download files whose names contain this keyword.")
    parser.add_argument("--list", action="store_true", help="Only list available files.")
    args = parser.parse_args()

    record = fetch_record()
    files = record.get("files", [])
    if args.list:
        for item in files:
            name = item.get("key", "")
            size = item.get("size", 0) / (1024 ** 3)
            print(f"{name}\t{size:.2f} GB")
        return

    selected = [item for item in files if args.contains.lower() in item.get("key", "").lower()]
    if not selected:
        raise SystemExit(f"No files matched keyword: {args.contains}")

    target = Path(args.target)
    for item in selected:
        name = item["key"]
        url = item["links"]["self"]
        download_file(url, target / name, item.get("size", 0))


if __name__ == "__main__":
    main()
