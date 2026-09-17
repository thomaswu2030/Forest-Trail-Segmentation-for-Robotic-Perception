from pathlib import Path
import shutil
import urllib.request

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(exist_ok=True)

base_url = "https://deepscene.cs.uni-freiburg.de/static/datasets/"
archive_name = "freiburg_forest_annotated.tar.gz"

parts = []

for suffix in ["aa", "ab", "ac"]:
    filename = f"{archive_name}.part-{suffix}"
    destination = data_dir / filename

    print(f"Downloading {filename}...", flush=True)
    urllib.request.urlretrieve(base_url + filename, destination)
    parts.append(destination)

print("Combining downloaded parts...", flush=True)

with (data_dir / archive_name).open("wb") as output:
    for part in parts:
        with part.open("rb") as source:
            shutil.copyfileobj(source, output)

print("Download complete.")