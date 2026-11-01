import argparse
import fnmatch
import io
import json
import shutil
import ssl
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zipfile_deflate64  # noqa: F401

DATA = Path(__file__).resolve().parents[1] / "data"
CHUNK = 2**20
URLS = {
    "cluster": "https://ndownloader.figshare.com/files/61118716",
    "lmt108": "https://zeus.lkn.ei.tum.de/downloads/texture/download/LMT_108_SurfaceMaterials_Database_V1.1.zip",
    "greatest_hits": "https://web.eecs.umich.edu/~ahowens/vis/vis-data.zip",
}
PATTERNS = {
    "cluster": ["*/sensor_data/audio/*", "*/sensor_data/accel/*", "*/texture_list.xlsx", "*/README.md"],
    "lmt108": ["*/AccelScansComponents/Movement/*", "*/SoundScans/Movement/*"],
}
SENS3_API = "https://data.4tu.nl/v2/articles/e7f8f6dd-b61c-42fc-bea9-5ff103cbe396/files"
HITS_CERTIFICATE = "http://crt.sectigo.com/InCommonRSAOVSSLCA3.crt"
DATASETS = ["cluster", "lmt108", "sens3", "greatest_hits"]


def open_url(request, context=None):
    for attempt in range(10):
        try:
            return urllib.request.urlopen(request, timeout=60, context=context)
        except OSError as error:
            if attempt == 9:
                raise
            print(f"{error}, retrying in {30 * (attempt + 1)} s", flush=True)
            time.sleep(30 * (attempt + 1))


class RemoteFile(io.RawIOBase):
    def __init__(self, url, context=None):
        self.context = context
        self.stream = None
        self.stream_position = None
        self.position = 0
        with open_url(urllib.request.Request(url, method="HEAD"), context) as head:
            self.url = head.url
            self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        for attempt in range(5):
            if self.stream_position != self.position:
                if self.stream is not None:
                    self.stream.close()
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-"})
                self.stream = open_url(request, self.context)
                if self.stream.status != 206:
                    raise OSError(f"Server does not support partial downloads: {self.url}")
                self.stream_position = self.position
            try:
                n_read = self.stream.readinto(buffer)
                break
            except OSError:
                if attempt == 4:
                    raise
                self.stream_position = None
                time.sleep(10)
        self.position += n_read
        self.stream_position = self.position
        return n_read

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def keep_file(dataset, member):
    name = Path(member.filename).name
    if member.is_dir():
        return False
    if dataset == "sens3":
        return "/Sliding_Friction-roughness/" in member.filename and (
            name.endswith(".wav") or name.startswith(("Material_sensor_", "Material_IR_pos_"))
        )
    if dataset == "greatest_hits":
        return name.endswith(("_denoised.wav", "_times.txt"))
    return any(fnmatch.fnmatchcase(member.filename, pattern) for pattern in PATTERNS[dataset])


def install_archive(dataset, url, context=None):
    with io.BufferedReader(RemoteFile(url, context), buffer_size=CHUNK) as remote, zipfile.ZipFile(remote) as archive:
        members = [member for member in archive.infolist() if keep_file(dataset, member)]
        if not members:
            raise ValueError(f"No {dataset} files found in {url}")
        for member in members:
            name = Path(member.filename)
            if dataset == "greatest_hits":
                name = Path(name.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Invalid archive path: {name}")
            path = DATA / dataset / name
            if dataset == "sens3" and name.name.startswith("Material_sensor_"):
                path = path.with_suffix(".npy")
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".part")
            print(f"{dataset}: {name}", flush=True)
            with archive.open(member) as source, open(partial, "wb") as target:
                if path.suffix == ".npy":
                    values = pd.read_csv(source, header=None, dtype=np.float64).to_numpy(np.float32)
                    np.save(target, values)
                else:
                    shutil.copyfileobj(source, target, CHUNK)
            partial.rename(path)


def install_dataset(dataset):
    marker = DATA / dataset / ".installed"
    if marker.exists():
        print(f"{dataset}: already installed")
        return
    if dataset == "sens3":
        with open_url(SENS3_API) as response:
            archives = [entry["download_url"] for entry in json.load(response) if entry["name"].startswith("Material")]
        with ThreadPoolExecutor(2) as pool:
            for _ in pool.map(lambda url: install_archive(dataset, url), archives):
                pass
    elif dataset == "greatest_hits":
        with open_url(HITS_CERTIFICATE) as response:
            certificate = response.read()
        context = ssl.create_default_context()
        context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(certificate))
        install_archive(dataset, URLS[dataset], context)
    else:
        install_archive(dataset, URLS[dataset])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"{dataset}: installed in {marker.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="all", choices=["all", *DATASETS])
    args = parser.parse_args()
    for dataset in DATASETS if args.dataset == "all" else [args.dataset]:
        install_dataset(dataset)
