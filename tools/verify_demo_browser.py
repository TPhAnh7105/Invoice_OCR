"""Browser integration check using real OCR; requires playwright and local Edge."""
import json
from io import BytesIO
from pathlib import Path
import time

from openpyxl import load_workbook
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/fastapi_demo/verification"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        assert page.title().startswith("Invoice Studio")
        page.screenshot(path=str(OUTPUT / "desktop-empty.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(OUTPUT / "mobile-empty.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1050})
        with page.expect_response(lambda response: response.url.endswith("/api/jobs") and response.request.method == "POST", timeout=30000) as event:
            page.locator("#sample-button").click()
        created = event.value.json()
        print("JOB", created["id"], flush=True)
        start = time.monotonic()
        job = created
        previous = None
        while time.monotonic() - start < 240:
            job = page.request.get(f"http://127.0.0.1:8000/api/jobs/{created['id']}").json()
            state = (job["stage"], job["progress"])
            if state != previous:
                print("PROGRESS", state, job["message"], flush=True)
                previous = state
            if job["status"] in {"done", "error"}:
                break
            page.wait_for_timeout(1000)
        (OUTPUT / "real-ocr-result.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        if job["status"] != "done":
            page.screenshot(path=str(OUTPUT / "failure.png"), full_page=True)
            raise AssertionError(job.get("message"))
        page.locator("#export-button").wait_for(state="visible")
        page.wait_for_function("!document.querySelector('#export-button').disabled", timeout=10000)
        selected = {index for field in job["result"]["fields"].values() for index in field["region_ids"]}
        assert page.locator(".region-poly").count() == len(selected)
        page.locator("#box-scope").select_option("all")
        assert page.locator(".region-poly").count() == job["result"]["region_count"]
        page.locator("#box-scope").select_option("fields")
        for region in job["result"]["regions"]:
            assert all(0 <= x <= job["width"] and 0 <= y <= job["height"] for x, y in region["box"])
        page.locator('[data-focus="TOTAL_COST"]').click()
        if job["result"]["fields"]["TOTAL_COST"]["region_ids"]:
            assert page.locator(".region-poly.focused").count() > 0
        page.locator("#clear-focus").click()
        page.locator("#show-boxes").uncheck()
        assert not page.locator("#overlay").is_visible()
        page.locator("#show-boxes").check()
        assert page.locator("#overlay").is_visible()
        page.locator("#expand").click()
        assert "expanded" in page.locator("#image-stage").get_attribute("class")
        page.locator("#expand").click()
        page.screenshot(path=str(OUTPUT / "desktop-result.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(OUTPUT / "mobile-result.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1050})
        original = page.locator("#field-SELLER").input_value()
        page.locator("#field-SELLER").fill("Cửa hàng kiểm thử Excel")
        with page.expect_download(timeout=20000) as event:
            page.locator("#export-button").click()
        event.value.save_as(str(OUTPUT / "verified-export.xlsx"))
        workbook = load_workbook(OUTPUT / "verified-export.xlsx")
        assert workbook["Hoa don"]["B2"].value == "Cửa hàng kiểm thử Excel"
        assert workbook["Vung chu"].max_row == len(job["result"]["regions"]) + 1
        # Restore original content; keep the exported verification file separate.
        page.request.patch(f"http://127.0.0.1:8000/api/jobs/{job['id']}/fields", data={"fields": {"SELLER": original}})
        # Exercise actual file input, empty detections, and reuse of loaded models.
        from PIL import Image
        blank = BytesIO()
        Image.new("RGB", (640, 480), "white").save(blank, "PNG")
        with page.expect_response(lambda response: response.url.endswith("/api/jobs") and response.request.method == "POST"):
            page.locator("#file-input").set_input_files({"name": "blank.png", "mimeType": "image/png", "buffer": blank.getvalue()})
        page.wait_for_function("document.querySelector('#result-badge').textContent === 'CHƯA TÌM THẤY CHỮ'", timeout=30000)
        assert page.locator(".region-poly").count() == 0
        assert all(page.locator(f"#field-{key}").input_value() == "" for key in ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"])
        assert not errors, errors
        print("VERIFIED", json.dumps({"regions": job["result"]["region_count"], "elapsed_seconds": job["result"]["elapsed_seconds"], "fields": {k: v["value"] for k, v in job["result"]["fields"].items()}, "browser_errors": errors}, ensure_ascii=False), flush=True)
        browser.close()


if __name__ == "__main__":
    main()
