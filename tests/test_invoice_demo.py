from io import BytesIO
import threading
import time

from fastapi.testclient import TestClient
from openpyxl import load_workbook
from PIL import Image

from invoice_demo.app import create_app
from invoice_demo.engine import LABELS, summarize


def photo(*, orientation=None):
    output = BytesIO()
    image = Image.new("RGB", (80, 40), "white")
    exif = image.getexif()
    if orientation:
        exif[274] = orientation
    image.save(output, "JPEG", exif=exif)
    return output.getvalue()


class Engine:
    loaded = True

    def __init__(self, gate=None, fail=False):
        self.gate = gate
        self.fail = fail

    def process(self, path, progress):
        if self.gate:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("Test failure")
        width, height = Image.open(path).size
        regions = [{"id": 0, "label": "SELLER", "text": "=1+1", "box": [[0, 0], [20, 0], [20, 10], [0, 10]], "ocr_confidence": .9, "classification_confidence": .8, "rule": False}]
        progress("classifying", 90, "Test", regions=regions)
        return {"width": width, "height": height, "regions": regions, "fields": summarize(regions), "elapsed_seconds": 0.1, "region_count": 1, "device": "cpu", "notice": ""}


def done(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/jobs/{job_id}").json()
        if result["status"] in {"done", "error"}:
            return result
        time.sleep(.01)
    raise AssertionError("Job did not finish")


def test_upload_orientation_edit_export_and_reload(tmp_path):
    with TestClient(create_app(engine=Engine(), data_dir=tmp_path)) as client:
        assert client.get("/").status_code == 200
        response = client.post("/api/jobs", files={"file": ("../hoá-đơn.jpg", photo(orientation=6), "image/jpeg")})
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = done(client, job_id)
        assert job["filename"] == "hoá-đơn.jpg"
        assert (job["width"], job["height"]) == (40, 80)
        image = Image.open(BytesIO(client.get(job["image_url"]).content))
        assert image.size == (40, 80)
        patched = client.patch(f"/api/jobs/{job_id}/fields", json={"fields": {"TOTAL_COST": "=HYPERLINK(\"x\")", "SELLER": "Cửa hàng Việt"}})
        assert patched.status_code == 200
        response = client.get(f"/api/jobs/{job_id}/excel")
        assert response.status_code == 200
        workbook = load_workbook(BytesIO(response.content))
        assert workbook.sheetnames == ["Hoa don", "Vung chu", "Doi chieu"]
        hoa_don_headers = [cell.value for cell in workbook["Hoa don"][1]]
        assert not any("tin cậy" in str(h).lower() or "confidence" in str(h).lower() for h in hoa_don_headers)
        assert workbook["Hoa don"]["B2"].value == "Cửa hàng Việt"
        assert workbook["Hoa don"]["E2"].data_type == "s"
        assert workbook["Vung chu"]["C2"].value == "=1+1"
        assert workbook["Vung chu"]["C2"].data_type == "s"
        assert workbook["Doi chieu"]["B2"].value == "=1+1"
        assert workbook["Doi chieu"]["C2"].value == "Cửa hàng Việt"
    with TestClient(create_app(engine=Engine(), data_dir=tmp_path)) as client:
        assert client.get(f"/api/jobs/{job_id}").json()["result"]["fields"]["SELLER"]["value"] == "Cửa hàng Việt"


def test_invalid_files_ids_and_fields(tmp_path):
    with TestClient(create_app(engine=Engine(), data_dir=tmp_path)) as client:
        assert client.post("/api/jobs", files={"file": ("bad.png", b"not an image", "image/png")}).status_code == 415
        assert client.post("/api/jobs", files={"file": ("big.png", b"x" * (15 * 1024 * 1024 + 1), "image/png")}).status_code == 413
        assert client.get("/api/jobs/not-a-uuid").status_code == 422
        assert client.get("/api/jobs/00000000-0000-0000-0000-000000000000").status_code == 404
        job_id = client.post("/api/jobs", files={"file": ("ok.jpg", photo(), "image/jpeg")}).json()["id"]
        done(client, job_id)
        assert client.patch(f"/api/jobs/{job_id}/fields", json={"fields": {"UNKNOWN": "x"}}).status_code == 422
        assert client.patch(f"/api/jobs/{job_id}/fields", json={"fields": {"SELLER": "x" * 4001}}).status_code == 422


def test_pending_job_does_not_allow_export_or_edit(tmp_path):
    gate = threading.Event()
    with TestClient(create_app(engine=Engine(gate=gate), data_dir=tmp_path)) as client:
        try:
            job_id = client.post("/api/jobs", files={"file": ("ok.jpg", photo(), "image/jpeg")}).json()["id"]
            assert client.get(f"/api/jobs/{job_id}/excel").status_code == 409
            assert client.patch(f"/api/jobs/{job_id}/fields", json={"fields": {"SELLER": "x"}}).status_code == 409
        finally:
            gate.set()
        assert done(client, job_id)["status"] == "done"


def test_failed_job_is_reported(tmp_path):
    with TestClient(create_app(engine=Engine(fail=True), data_dir=tmp_path)) as client:
        job_id = client.post("/api/jobs", files={"file": ("ok.jpg", photo(), "image/jpeg")}).json()["id"]
        assert done(client, job_id)["status"] == "error"


def test_empty_fields_are_not_fabricated():
    fields = summarize([])
    assert set(fields) == set(LABELS)
    assert all(field["value"] == "" and field["region_ids"] == [] for field in fields.values())
    # Unextracted fields must have 0.0% confidence
    assert all(field["confidence"] == 0.0 for field in fields.values())
    assert all(field["confidence_display"] == "0%" for field in fields.values())


def test_confidence_and_overall_mean():
    from invoice_demo.engine import compute_overall_confidence
    def region(index, text, box, label, ocr_c=0.9, cls_c=0.8):
        return {"id": index, "text": text, "box": box, "label": label, "ocr_confidence": ocr_c, "classification_confidence": cls_c, "rule": False}

    # Only SELLER extracted (0.9 * 0.8 = 0.72 -> 72.0%), others empty (0%)
    regions = [region(0, "VinCommerce", [[0, 0], [40, 0], [40, 10], [0, 10]], "SELLER")]
    fields = summarize(regions)
    assert fields["SELLER"]["confidence"] == 72.0
    assert fields["SELLER"]["confidence_display"] == "72.0%"
    assert fields["ADDRESS"]["confidence"] == 0.0
    assert fields["TIMESTAMP"]["confidence"] == 0.0
    assert fields["TOTAL_COST"]["confidence"] == 0.0

    # Overall mean = (72.0 + 0 + 0 + 0) / 4 = 18.0%
    overall = compute_overall_confidence(fields)
    assert overall == 18.0


def test_total_amount_joins_nearby_numeric_box():
    def region(index, text, box, label):
        return {"id": index, "text": text, "box": box, "label": label, "ocr_confidence": .9, "classification_confidence": .9, "rule": label == "TOTAL_COST"}
    regions = [region(0, "Tổng tiền", [[0, 0], [40, 0], [40, 10], [0, 10]], "TOTAL_COST"), region(1, "125.000", [[50, 0], [90, 0], [90, 10], [50, 10]], "ADDRESS")]
    assert summarize(regions)["TOTAL_COST"]["value"] == "Tổng tiền 125.000"


def test_history_listing_and_reload_from_disk(tmp_path):
    with TestClient(create_app(engine=Engine(), data_dir=tmp_path)) as client:
        # Initially empty history
        res0 = client.get("/api/jobs")
        assert res0.status_code == 200
        assert res0.json() == []

        # Upload first invoice
        resp1 = client.post("/api/jobs", files={"file": ("invoice-1.jpg", photo(), "image/jpeg")})
        job1_id = resp1.json()["id"]
        done(client, job1_id)

        # Upload second invoice
        resp2 = client.post("/api/jobs", files={"file": ("invoice-2.jpg", photo(), "image/jpeg")})
        job2_id = resp2.json()["id"]
        done(client, job2_id)

        # List history: should return 2 jobs, sorted newest first
        res_history = client.get("/api/jobs").json()
        assert len(res_history) == 2
        ids = [item["id"] for item in res_history]
        assert ids == [job2_id, job1_id]

        # Verify each job's image can be loaded independently
        img1 = client.get(f"/api/jobs/{job1_id}/image")
        assert img1.status_code == 200
        assert img1.headers["content-type"] == "image/jpeg"

        img2 = client.get(f"/api/jobs/{job2_id}/image")
        assert img2.status_code == 200
        assert img2.headers["content-type"] == "image/jpeg"

        # Verify detail retrieval for each history item
        detail1 = client.get(f"/api/jobs/{job1_id}").json()
        assert detail1["filename"] == "invoice-1.jpg"
        assert detail1["status"] == "done"

        detail2 = client.get(f"/api/jobs/{job2_id}").json()
        assert detail2["filename"] == "invoice-2.jpg"
        assert detail2["status"] == "done"

    # Simulate restarting the server (new client with empty in-memory state on same data_dir)
    with TestClient(create_app(engine=Engine(), data_dir=tmp_path)) as client:
        reloaded = client.get("/api/jobs").json()
        assert len(reloaded) == 2
        reloaded_ids = [item["id"] for item in reloaded]
        assert job1_id in reloaded_ids and job2_id in reloaded_ids

