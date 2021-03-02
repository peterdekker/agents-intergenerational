
import os
import pathlib
import requests
import shutil

DATA_ARCHIVE_URL = "https://github.com/lessersunda/lexirumah-data/archive/v3.0.0.tar.gz"
DATA_ARCHIVE_PATH = "lexirumah.tar.gz"
DATA_UNPACK_PATH = "lexirumah-data-3.0.0"
DATA_PATH = os.path.join(DATA_UNPACK_PATH, "cldf/forms.csv")
LECTS_PATH = os.path.join(DATA_UNPACK_PATH, "cldf/lects.csv")


def download_if_needed(file_path, url, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        p = pathlib.Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            print(f"Downloading {label} from {url}")
            try:
                r = requests.get(url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if file_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(file_path)


def download_lexirumah():
    download_if_needed(DATA_ARCHIVE_PATH, DATA_ARCHIVE_URL, "LexiRumah")