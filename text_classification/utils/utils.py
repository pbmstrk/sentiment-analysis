import gzip
import os
import shutil
from typing import Optional

import requests
from tqdm import tqdm


def _process_resp(resp, filepath):
    chunk_size = 16 * 1024
    total_size = int(resp.headers.get("Content-length", 0))
    with open(filepath, "wb") as outfile:
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for data in resp.iter_content(chunk_size):
                if data:
                    outfile.write(data)
                    pbar.update(len(data))


def download_file(
    url: str,
    name: str,
    root: str = ".data",
    filename: Optional[str] = None,
    override: bool = False,
):

    # File will be stored in root/name/filename
    # root and name are passed as parameters
    # filename is inferred from url

    # create directory if it doesn't exist
    if not os.path.exists(os.path.join(root, name)):
        os.makedirs(os.path.join(root, name))

    if not filename:
        _, filename = os.path.split(url)

    filepath = os.path.join(root, name, filename)

    if not os.path.exists(filepath) or override:
        print(f"Downloading {filename} from {url}.\nSaving to:{filepath}")

        if not "drive.google.com" in url:
            resp = requests.get(url, stream=True)
            _process_resp(resp, filepath)
        else:
            # from https://github.com/pytorch/text/blob/master/torchtext/utils.py#L121-L129
            confirm_token = None
            session = requests.Session()
            resp = session.get(url, stream=True)
            for k, v in resp.cookies.items():
                if k.startswith("download_warning"):
                    confirm_token = v

            if confirm_token:
                url = url + "&confirm=" + confirm_token
                resp = session.get(url, stream=True)

            _process_resp(resp, filepath)
    else:
        print(f"Found file at {os.path.join(root, name)} skipping download.")
        return

    return filepath


def extract_from_file(zfile: str, drc: str):

    # only extracts if download returns filename
    if not zfile:
        return

    # special case word word2vec
    if zfile.endswith("bin.gz"):
        print(f"Extracting: {zfile}")
        filepath = zfile[:-3]
        with gzip.open(zfile, "rb") as f_in:
            with open(filepath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    else:
        shutil.unpack_archive(zfile, drc)


def download_extract(
    url: str,
    name: str,
    root: str = ".data",
    override: bool = False,
    filename: Optional[str] = None,
    extract_only: Optional[bool] = False,
):
    extract_dir = os.path.join(root, name)

    if extract_only:
        extract_from_file(os.path.join(root, name, filename), os.path.join(root, name))
    else:
        zfile = download_file(
            url, name, root=root, override=override, filename=filename
        )
        extract_from_file(zfile, extract_dir)
