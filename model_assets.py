"""Download the public model assets used by invoice-ocr.ipynb, never datasets."""

import csv
import hashlib
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


SOURCE_REPOSITORY = "https://github.com/kaylode/vietnamese-ocr-toolbox"
SOURCE_COMMIT = "b4d6cb26f2397bb30ee639b5a5ea1d187a69e445"
RAW_BASE = f"https://raw.githubusercontent.com/kaylode/vietnamese-ocr-toolbox/{SOURCE_COMMIT}"
CONFIG_BASE_URL = f"{RAW_BASE}/tool/config/ocr/models"

# Public checkpoints published by the original project. These replace three
# unavailable private links in the imported notebook, not the local image data.
WEIGHT_IDS = {
    "pan_resnet18_sroie19": "1-Nj8TSM_eqZDZzRArZjWPcVCtl1l2uQP",
    "transformerocr_mcocr": "1qpXp_-digz2HPTGY_GPdwstzGLhjC_ot",
    "phobert_mcocr": "1v4GQPg4Jx5FWvqJ-2k9YCxEd6iFdlXXa",
}


def download_url(source):
    """Convert a public Drive ID/share link to the file-content endpoint."""
    if not source.startswith(("http://", "https://")):
        file_id = source
    else:
        parsed = urlparse(source)
        if parsed.hostname not in {"drive.google.com", "drive.usercontent.google.com"}:
            return source
        file_id = parse_qs(parsed.query).get("id", [None])[0]
        if file_id is None and "/file/d/" in parsed.path:
            file_id = parsed.path.split("/file/d/", 1)[1].split("/", 1)[0]
        if file_id is None:
            raise ValueError(f"Cannot identify Google Drive file: {source}")
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def validate_asset(path, kind):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Empty or missing file: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(256)
    if prefix.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError(f"Received a web page instead of a model asset: {path.name}")
    if kind == "torch":
        if path.stat().st_size < 1_000_000 or not prefix.startswith((b"PK\x03\x04", b"\x80")):
            raise ValueError(f"Not a complete PyTorch checkpoint: {path.name}")
    elif kind == "csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns = csv.DictReader(stream).fieldnames or []
        if not {"text", "lbl"}.issubset(columns):
            raise ValueError(f"Dictionary must contain text/lbl columns: {path.name}")
    return path


def ensure_asset(source, target, *, kind="text", allow_download=True, md5=None):
    """Reuse valid local files; publish a download only after validation."""
    target = Path(target).resolve()
    try:
        validate_asset(target, kind)
        if md5 and hashlib.md5(target.read_bytes()).hexdigest() != md5:
            raise ValueError(f"MD5 mismatch: {target.name}")
        return target
    except (OSError, ValueError):
        if not allow_download:
            raise FileNotFoundError(
                f"Missing/invalid local model asset: {target}\n"
                f"Manual download: {download_url(source)}"
            ) from None

    target.parent.mkdir(parents=True, exist_ok=True)
    url = download_url(source)
    temporary = None
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError("Server returned a login/permission page instead of the file")
            expected = response.headers.get("Content-Length")
            total = 0
            next_progress = 50 * 1024 * 1024
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=target.name + ".", suffix=".download", delete=False
            ) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    total += len(chunk)
                    if total >= next_progress:
                        print(f"{target.name}: {total / (1024 * 1024):.0f} MiB", flush=True)
                        next_progress += 50 * 1024 * 1024
            if expected is not None and total != int(expected):
                raise ValueError(f"Incomplete transfer: {total}/{expected} bytes")
        validate_asset(temporary, kind)
        if md5 and hashlib.md5(temporary.read_bytes()).hexdigest() != md5:
            raise ValueError(f"MD5 mismatch: {target.name}")
        temporary.replace(target)
        print(f"Saved {target.name} ({total / (1024 * 1024):.1f} MiB)", flush=True)
        return target
    except Exception as error:
        raise RuntimeError(
            f"Cannot prepare {target.name}: {error}\n"
            f"Manual download: {url}\nSave as: {target}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_model_assets(project, *, allow_download=True):
    project = Path(project).resolve()
    assets = [
        ("weights/PANNet_best_map.pth", WEIGHT_IDS["pan_resnet18_sroie19"], "torch"),
        ("weights/transformerocr.pth", WEIGHT_IDS["transformerocr_mcocr"], "torch"),
        ("weights/bert_weight.pth", WEIGHT_IDS["phobert_mcocr"], "torch"),
        ("configs/detection/configs.yaml", f"{RAW_BASE}/tool/config/detection/configs.yaml", "text"),
        ("configs/ocr/configs.yaml", f"{RAW_BASE}/tool/config/ocr/configs.yaml", "text"),
        ("heuristic/custom-dictionary.csv", f"{RAW_BASE}/modules/retrieval/heuristic/custom-dictionary.csv", "csv"),
    ]
    for name in (
        "base.yml", "vgg-transformer.yml", "resnet-transformer.yml",
        "resnet_fpn_transformer.yml", "vgg-seq2seq.yml", "vgg-convseq2seq.yml",
    ):
        assets.append((f"configs/ocr/models/{name}", f"{CONFIG_BASE_URL}/{name}", "text"))
    prepared, errors = {}, []
    for relative, url, kind in assets:
        try:
            prepared[relative] = ensure_asset(
                url, project / relative, kind=kind, allow_download=allow_download
            )
        except (OSError, RuntimeError) as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError(
            f"{len(prepared)}/{len(assets)} model assets ready. Remaining files:\n\n"
            + "\n\n".join(errors)
        )
    print(f"Model assets ready: {len(prepared)}/{len(assets)}. Local dataset unchanged.")
    return prepared
