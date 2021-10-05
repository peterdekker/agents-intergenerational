
from pyclts import CLTS
import os
import requests
import shutil

verb_forms = {"suffixing": ["balik", "bu'a", "buka", "bəsuk", "de'in", "deka", "gehi", "gelu", "(gə)redo", "gəta", "haga", "həbo", "hitun", "hode", "horon", "kantar", "kirin", "kədoko", "kərian", "koda", "ləba wəkin", "lodo", "louk", "mia", "mori", "nyanyi", "ola", "peko", "peun", "pəla'e", "pupu", "səga", "taku", "tanin", "tei", "tobo", "tor", "turu", "tutu"],
              "prefixing": ["an", "a'an", "əte", "a'i", "enun", "oi", "ala", "ələ", "anan", "ahu' wai", "awa", "awan", "ian", "əwan", "itə", "iu", "odi", "olin", "urən"]}

affixes = {"prefixing": ["k", "m", "n", "m", "t", "m", "r"],
           "suffixing": ["kən", "ko", "no", "na", "kən", "te", "ke", "ne", "ka"]}

currentdir = os.path.dirname(os.path.realpath(__file__))
CLTS_ARCHIVE_PATH = os.path.join(currentdir, "2.1.0.tar.gz")
CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
CLTS_PATH = os.path.join(currentdir, "clts-2.1.0")


def download_if_needed(archive_path, archive_url, file_path, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        #p = pathlib.Path(file_path)
        #p.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, 'wb') as f:
            print(f"Downloading {label} from {archive_url}")
            try:
                r = requests.get(archive_url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if archive_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(archive_path, currentdir)


def load_clts():
    # Download CLTS
    download_if_needed(CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH, "CLTS")
    return CLTS(CLTS_PATH)


clts = load_clts()
for vt in ["prefixing", "suffixing"]:
    print("prefixing")
    for form in verb_forms[vt]:
        for aff in affixes[vt]:
            inflected_form = aff+form if vt == "prefixing" else form+aff
            print(inflected_form)
            spaced_form = " ".join(list(inflected_form))
            cv_pattern = clts.bipa.translate(spaced_form, clts.soundclass("cv")).replace(" ", "")
            print(cv_pattern)
            if "CC" in cv_pattern:
                print("CONSONANT CLUSTER!")
