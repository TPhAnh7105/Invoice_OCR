"""FastAPI upload, background OCR jobs, editable fields and Excel export."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import threading
from uuid import UUID, uuid4
import warnings

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .engine import FIELD_NAMES, LABELS, InvoiceEngine, ROOT

LOGGER = logging.getLogger("invoice_demo")
MAX_UPLOAD = 15 * 1024 * 1024
MAX_PIXELS = 24_000_000
STATIC = Path(__file__).parent / "static"


class FieldEdit(BaseModel):
    fields: dict[str, str] = Field(default_factory=dict)


def excel_text(cell, value):
    """Write untrusted OCR as literal strings, including values starting with '='."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    cell.value = ILLEGAL_CHARACTERS_RE.sub("", str(value))[:32767]
    cell.data_type = "s"


def make_excel(job):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    book = Workbook()
    sheet = book.active
    sheet.title = "Hoa don"
    result = job["result"]
    headers = ["Tên tệp", *[FIELD_NAMES[key] for key in LABELS], "Ngày xử lý (UTC)", "Thời gian xử lý (giây)"]
    sheet.append(headers)
    values = [job["filename"], *[result["fields"][key]["value"] for key in LABELS], job["created_at"], str(result["elapsed_seconds"])]
    for col, value in enumerate(values, 1):
        excel_text(sheet.cell(2, col), value)
    sheet.row_dimensions[2].height = 85
    details = book.create_sheet("Vung chu")
    details.append(["Vùng", "Nhãn", "Văn bản OCR", "Độ tin cậy OCR", "Điểm PhoBERT", "Nhãn được sửa bởi luật", "Tọa độ bốn góc (pixel ảnh)"])
    for index, region in enumerate(result["regions"], 2):
        details.cell(index, 1, region["id"] + 1)
        for col, value in ((2, region["label"]), (3, region["text"]), (7, json.dumps(region["box"], ensure_ascii=False))):
            excel_text(details.cell(index, col), value)
        details.cell(index, 4, region["ocr_confidence"]).number_format = "0.0%"
        details.cell(index, 5, region["classification_confidence"]).number_format = "0.0%"
        details.cell(index, 6, bool(region.get("rule")))
    original = book.create_sheet("Doi chieu")
    original.append(["Trường", "Kết quả ban đầu", "Kết quả xuất Excel"])
    for index, key in enumerate(LABELS, 2):
        field = result["fields"][key]
        for col, value in enumerate((FIELD_NAMES[key], field.get("original_value", field["value"]), field["value"]), 1):
            excel_text(original.cell(index, col), value)
    for ws in book:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="174D3C")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 32
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(col)].width = 30 if col != 3 else 54
    data = BytesIO()
    book.save(data)
    data.seek(0)
    return data


class Jobs:
    def __init__(self, root, engine):
        self.root = root
        self.engine = engine
        self.lock = threading.RLock()
        self.records = {}
        self.slots = threading.BoundedSemaphore(4)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="invoice-ocr")

    def update(self, job_id, **changes):
        with self.lock:
            self.records[job_id].update(deepcopy(changes))

    def get(self, job_id):
        with self.lock:
            if job_id in self.records:
                return deepcopy(self.records[job_id])
            path = self.root / job_id / "result.json"
            if path.is_file():
                self.records[job_id] = json.loads(path.read_text(encoding="utf-8"))
                return deepcopy(self.records[job_id])
        raise HTTPException(404, "Không tìm thấy phiên đọc ảnh này.")

    def list(self, limit=50):
        items = []
        with self.lock:
            known = set(self.records.keys())
            if self.root.is_dir():
                for entry in self.root.iterdir():
                    if entry.is_dir() and entry.name not in known:
                        path = entry / "result.json"
                        if path.is_file():
                            try:
                                self.records[entry.name] = json.loads(path.read_text(encoding="utf-8"))
                                known.add(entry.name)
                            except Exception:
                                pass
            for job_id, record in self.records.items():
                res = record.get("result") or {}
                fields = res.get("fields") or {}
                items.append({
                    "id": job_id,
                    "filename": record.get("filename", "hoa-don.jpg"),
                    "created_at": record.get("created_at", ""),
                    "status": record.get("status", "unknown"),
                    "stage": record.get("stage", ""),
                    "image_url": record.get("image_url", f"/api/jobs/{job_id}/image"),
                    "seller": fields.get("SELLER", {}).get("value", ""),
                    "total_cost": fields.get("TOTAL_COST", {}).get("value", ""),
                    "overall_confidence": res.get("overall_confidence", 0.0),
                    "overall_confidence_display": res.get("overall_confidence_display", "0%"),
                    "region_count": res.get("region_count", len(record.get("regions", []))),
                })
        items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
        return items[:limit]

    def save(self, job_id):
        with self.lock:
            path = self.root / job_id / "result.json"
            pending = path.with_suffix(".tmp")
            pending.write_text(json.dumps(self.records[job_id], ensure_ascii=False, indent=2), encoding="utf-8")
            pending.replace(path)

    def run(self, job_id):
        def progress(stage, percent, message, **extra):
            self.update(job_id, status="processing", stage=stage, progress=percent, message=message, **extra)

        try:
            result = self.engine.process(self.root / job_id / "image.jpg", progress)
            for field in result["fields"].values():
                field["original_value"] = field["value"]
            self.update(job_id, status="done", stage="done", progress=100, message="Đã đọc xong hóa đơn.", result=result, regions=result["regions"])
            self.save(job_id)
        except Exception as exc:
            LOGGER.exception("OCR job %s failed", job_id)
            message = str(exc)
            if "out of memory" in message.lower():
                message = "GPU không đủ bộ nhớ. Đóng tác vụ OCR khác hoặc khởi động demo với OCR_DEVICE=cpu."
            elif "cached files" in message or "offline" in message.lower() or "outgoing traffic" in message.lower():
                message = "Chưa có cấu hình/tokenizer PhoBERT trong cache. Chạy demo với OCR_ALLOW_DOWNLOAD=1 để chuẩn bị lần đầu."
            self.update(job_id, status="error", stage="error", message="Không đọc được ảnh: " + message[:600])
            self.save(job_id)
        finally:
            self.slots.release()


def create_app(*, engine=None, data_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        root = Path(data_dir or os.getenv("OCR_OUTPUT_DIR", ROOT / "output" / "fastapi_demo")).resolve()
        root.mkdir(parents=True, exist_ok=True)
        app.state.jobs = Jobs(root, engine or InvoiceEngine())
        yield
        app.state.jobs.executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Invoice Studio", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.jobs.engine.loaded}

    @app.get("/api/sample")
    def sample():
        for directory in (ROOT / "val_images/val_images", ROOT / "val_images"):
            path = directory / "mcocr_val_145114gpkup.jpg"
            if path.is_file():
                return FileResponse(path, media_type="image/jpeg")
        raise HTTPException(404, "Chưa có ảnh mẫu trong thư mục validation.")

    @app.post("/api/jobs", status_code=202)
    def upload(file: UploadFile = File(...)):
        jobs = app.state.jobs
        if not jobs.slots.acquire(blocking=False):
            file.file.close()
            raise HTTPException(429, "Đang có nhiều ảnh chờ xử lý. Vui lòng thử lại sau.")
        submitted = False
        try:
            content = file.file.read(MAX_UPLOAD + 1)
            if len(content) > MAX_UPLOAD:
                raise HTTPException(413, "Ảnh vượt quá 15 MB. Vui lòng chọn ảnh nhỏ hơn.")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(content)) as source:
                        if source.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                            raise HTTPException(415, "Hỗ trợ ảnh JPG, PNG, WEBP, BMP hoặc TIFF.")
                        if source.width * source.height > MAX_PIXELS:
                            raise HTTPException(413, "Ảnh vượt quá 24 triệu pixel. Hãy giảm độ phân giải.")
                        source.load()
                        oriented = ImageOps.exif_transpose(source)
                        rgba = oriented.convert("RGBA")
                        image = Image.new("RGB", rgba.size, "white")
                        image.paste(rgba, mask=rgba.getchannel("A"))
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                raise HTTPException(415, "File không phải ảnh hợp lệ hoặc đã bị hỏng.") from None
            job_id = str(uuid4())
            folder = jobs.root / job_id
            folder.mkdir()
            image.save(folder / "image.jpg", "JPEG", quality=95)
            filename = re.split(r"[/\\]", file.filename or "hoa-don.jpg")[-1][:180]
            record = {
                "id": job_id, "filename": filename, "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "queued", "stage": "queued", "progress": 0,
                "message": "Ảnh đã được nhận. Đang chờ đến lượt đọc…",
                "width": image.width, "height": image.height,
                "image_url": f"/api/jobs/{job_id}/image", "regions": [], "result": None,
            }
            with jobs.lock:
                jobs.records[job_id] = record
            jobs.executor.submit(jobs.run, job_id)
            submitted = True
            return jobs.get(job_id)
        finally:
            file.file.close()
            if not submitted:
                jobs.slots.release()

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50):
        return app.state.jobs.list(limit=limit)

    @app.get("/api/jobs/{job_id}")
    def status(job_id: UUID):
        return app.state.jobs.get(str(job_id))

    @app.get("/api/jobs/{job_id}/image")
    def image(job_id: UUID):
        app.state.jobs.get(str(job_id))
        return FileResponse(app.state.jobs.root / str(job_id) / "image.jpg", media_type="image/jpeg")

    @app.patch("/api/jobs/{job_id}/fields")
    def edit_fields(job_id: UUID, body: FieldEdit):
        jobs = app.state.jobs
        key = str(job_id)
        if set(body.fields) - set(LABELS) or any(len(value) > 4000 for value in body.fields.values()):
            raise HTTPException(422, "Trường không hợp lệ hoặc nội dung dài quá 4.000 ký tự.")
        with jobs.lock:
            job = jobs.get(key)
            if job["status"] != "done":
                raise HTTPException(409, "Hãy chờ đọc xong ảnh trước khi chỉnh sửa.")
            for label, value in body.fields.items():
                jobs.records[key]["result"]["fields"][label]["value"] = value
                if value.strip():
                    jobs.records[key]["result"]["fields"][label]["confidence"] = 100.0
                    jobs.records[key]["result"]["fields"][label]["confidence_display"] = "100.0%"
                else:
                    jobs.records[key]["result"]["fields"][label]["confidence"] = 0.0
                    jobs.records[key]["result"]["fields"][label]["confidence_display"] = "0%"
            fields_dict = jobs.records[key]["result"]["fields"]
            overall = round(sum(fields_dict[l].get("confidence", 0.0) for l in LABELS) / len(LABELS), 1)
            jobs.records[key]["result"]["overall_confidence"] = overall
            jobs.records[key]["result"]["overall_confidence_display"] = f"{overall:.1f}%"
            jobs.save(key)
        return jobs.get(key)

    @app.get("/api/jobs/{job_id}/excel")
    def export(job_id: UUID):
        job = app.state.jobs.get(str(job_id))
        if job["status"] != "done":
            raise HTTPException(409, "Excel sẽ sẵn sàng sau khi đọc xong ảnh.")
        return StreamingResponse(make_excel(job), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f'attachment; filename="hoa-don-{str(job_id)[:8]}.xlsx"'})

    return app


app = create_app()
