"""Test History UI switching with Playwright and mock Engine."""
import json
import threading
import time
from io import BytesIO
from pathlib import Path

import uvicorn
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invoice_demo.app import create_app
from invoice_demo.engine import LABELS, summarize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/fastapi_demo/test_history"


def make_image(text, color):
    img = Image.new("RGB", (300, 400), color)
    buf = BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class MockEngine:
    loaded = True

    def __init__(self):
        self.call_count = 0

    def process(self, path, progress):
        self.call_count += 1
        is_second = (self.call_count % 2 == 0)
        seller_name = "QUÁN CÀ PHÊ B" if is_second else "SIÊU THỊ A"
        total_str = "88.000" if is_second else "25.000"
        
        regions = [
            {"id": 0, "label": "SELLER", "text": seller_name, "box": [[10, 10], [200, 10], [200, 40], [10, 40]], "ocr_confidence": 0.95, "classification_confidence": 0.99, "rule": False},
            {"id": 1, "label": "ADDRESS", "text": "123 Đường ABC, Hà Nội", "box": [[10, 50], [250, 50], [250, 80], [10, 80]], "ocr_confidence": 0.92, "classification_confidence": 0.95, "rule": False},
            {"id": 2, "label": "TIMESTAMP", "text": "29/10/2026 12:30", "box": [[10, 90], [180, 90], [180, 110], [10, 110]], "ocr_confidence": 0.96, "classification_confidence": 0.98, "rule": True},
            {"id": 3, "label": "TOTAL_COST", "text": f"Tổng cộng {total_str}", "box": [[10, 150], [220, 150], [220, 180], [10, 180]], "ocr_confidence": 0.94, "classification_confidence": 0.97, "rule": True},
        ]
        progress("classifying", 100, "Done", regions=regions)
        return {
            "width": 300,
            "height": 400,
            "regions": regions,
            "fields": summarize(regions),
            "elapsed_seconds": 0.05,
            "region_count": len(regions),
            "device": "cpu",
            "notice": "",
        }


def run_server(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    port = 8765
    data_dir = OUTPUT / "server_data"
    import shutil
    if data_dir.is_dir():
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    app = create_app(engine=MockEngine(), data_dir=data_dir)
    server_thread = threading.Thread(target=run_server, args=(app, port), daemon=True)
    server_thread.start()
    time.sleep(1.2)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
        print("Page loaded")

        # 1. Upload invoice-a
        img_a = make_image("A", (240, 240, 255))
        page.locator("#file-input").set_input_files({"name": "invoice-a.jpg", "mimeType": "image/jpeg", "buffer": img_a})
        page.wait_for_function("document.querySelector('#result-badge').textContent === 'ĐÃ NHẬN DẠNG'", timeout=15000)
        
        # Verify invoice-a results
        assert page.locator("#field-SELLER").input_value() == "SIÊU THỊ A"
        assert "invoice-a.jpg" in page.locator("#receipt-image").get_attribute("src") or "/image" in page.locator("#receipt-image").get_attribute("src")
        
        # History section should be visible with 1 item
        page.wait_for_selector(".history-card", timeout=5000)
        assert page.locator(".history-card").count() == 1
        page.screenshot(path=str(OUTPUT / "step1_invoice_a.png"))
        print("Invoice A processed and visible in history")

        # 2. Upload invoice-b
        img_b = make_image("B", (255, 240, 240))
        page.locator("#file-input").set_input_files({"name": "invoice-b.jpg", "mimeType": "image/jpeg", "buffer": img_b})
        page.wait_for_function("document.querySelector('#field-SELLER').value === 'QUÁN CÀ PHÊ B'", timeout=15000)
        
        # History should now have 2 items
        page.wait_for_function("document.querySelectorAll('.history-card').length === 2", timeout=5000)
        assert page.locator(".history-card").count() == 2
        assert page.locator("#filename").text_content() == "invoice-b.jpg"
        assert page.locator("#field-SELLER").input_value() == "QUÁN CÀ PHÊ B"
        page.screenshot(path=str(OUTPUT / "step2_invoice_b.png"))
        print("Invoice B processed, 2 items in history")

        # 3. CLICK ON INVOICE A IN HISTORY!
        # History items are ordered newest first, so invoice-a is the 2nd card (index 1)
        card_a = page.locator(".history-card").nth(1)
        card_a.click()

        # Check that UI switches back to Invoice A:
        # - #filename becomes invoice-a.jpg
        # - #field-SELLER becomes SIÊU THỊ A
        # - card_a gets class 'active'
        page.wait_for_function("document.querySelector('#filename').textContent === 'invoice-a.jpg'", timeout=5000)
        page.wait_for_function("document.querySelector('#field-SELLER').value === 'SIÊU THỊ A'", timeout=5000)
        
        src_a = page.locator("#receipt-image").get_attribute("src")
        print(f"Switched image source: {src_a}")
        assert "/image" in src_a
        assert "active" in card_a.get_attribute("class")
        
        page.screenshot(path=str(OUTPUT / "step3_switched_to_a.png"))
        print("Successfully clicked history and switched to Invoice A!")

        # 4. CLICK ON INVOICE B IN HISTORY!
        card_b = page.locator(".history-card").nth(0)
        card_b.click()
        page.wait_for_function("document.querySelector('#filename').textContent === 'invoice-b.jpg'", timeout=5000)
        page.wait_for_function("document.querySelector('#field-SELLER').value === 'QUÁN CÀ PHÊ B'", timeout=5000)
        
        page.screenshot(path=str(OUTPUT / "step4_switched_to_b.png"))
        print("Successfully clicked history and switched back to Invoice B!")

        assert not errors, f"Page errors: {errors}"
        browser.close()
        print("ALL HISTORY TESTS PASSED!")


if __name__ == "__main__":
    main()
