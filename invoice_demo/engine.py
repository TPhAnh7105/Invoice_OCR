"""Persistent inference engine using the notebook's PAN, VietOCR and PhoBERT."""
from pathlib import Path
import os
import re
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST")
FIELD_NAMES = {
    "SELLER": "Tên cửa hàng", "ADDRESS": "Địa chỉ",
    "TIMESTAMP": "Ngày / giờ", "TOTAL_COST": "Tổng tiền",
}
DATE_PATTERN = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")
TIME_OF_DAY_PATTERN = re.compile(r"\b\d{1,2}\s*[:.]\s*\d{2}(?:\s*[:.]\s*\d{2})?\b")
TIME_PATTERN = re.compile(
    r"(?:\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{1,2}\s*[:.]\s*\d{2}(?:\s*[:.]\s*\d{2})?\b)",
    re.IGNORECASE
)
TOTAL_PATTERN = re.compile(
    r"\b(?:tong(?: cong| tien| thanh toan)?|thanh toan|thanh tien|cong tien(?: hang)?|tien hang|phai tra|phai t\.toan|t\.toan|total|grand total)\b",
    re.IGNORECASE
)
NON_TOTAL_PATTERN = re.compile(
    r"\b(?:tien thua|tra lai|tien thoi|giam gia|chiet khau|khuyen mai|tiet kiem|voucher|diem tich luy|don gia|so luong|mat hang)\b",
    re.IGNORECASE
)
AMOUNT_PATTERN = re.compile(
    r"(?:\d{1,3}(?:[., ]\d{3})+|\d{3,9})(?:[.,]\d{1,2})?(?=\s*(?:đ|d|vnd|vnđ|\b|$))",
    re.IGNORECASE
)

ADDRESS_INDICATORS = [
    "duong", "pho", "ngo", "ngach", "hem", "to ", "to.", "khu ", "thon ", "xom ",
    "phuong", "p.", "xa ", "quan ", "q.", "qg", "huyen ", "tp.", "tp ", "thanh pho",
    "tinh ", "dc:", "d/c:", "dia chi", "kdc", "tt.", "toa nha", "ha noi", "quang ninh",
    "ho chi minh", "tphcm", "da nang", "hai phong", "can tho", "cam pha", "gia lam", "go vap"
]

ADDRESS_DISQUALIFIERS = [
    "thu ngan", "cashier", "hotline", "website", "cam on", "chiet khau",
    "the khach hang", "gio vao", "gio ra", "wifi", "mat khau", "quay", "diem tich luy",
    "hoa don", "ban hang", "tra lai hang", "phieu", "order", "khach hang", "phuong thuc thanh toan",
    "banh mi", "nuoc uong", "pororo", "tam bong"
]

SELLER_DISQUALIFIERS = [
    "tax invoice", "hoa don", "phieu thanh toan", "cam on", "quy khach",
    "hotline", "website", "thu ngan", "cashier", "tam bong", "tay mg",
    "chiet khau", "nvbh", "tel:", "tei:", "dt:", "giao dich", "bien lai"
]

STORE_DICT = set()
try:
    dict_file = ROOT / "data/heuristic/custom-dictionary.csv"
    if dict_file.is_file():
        import csv
        with open(dict_file, encoding="utf-8") as _df:
            for _row in csv.DictReader(_df):
                if int(_row.get("lbl", -1)) == 0:
                    _t = str(_row.get("text", "")).strip().lower()
                    if len(_t) >= 3:
                        STORE_DICT.add(_t)
except Exception:
    pass


def amount_only(text):
    """Accept an amount crop, not an arbitrary number in product/date text."""
    text = re.sub(r"(?:vnd|vnđ|dong|đ|₫)", "", fold(text)).strip(" :;=-")
    return bool(re.fullmatch(r"\d{1,3}(?:[., ]\d{3})+(?:[.,]\d{1,2})?|\d{3,9}", text))


def total_keyword(text):
    text = fold(text)
    return bool(TOTAL_PATTERN.search(text)) and not bool(NON_TOTAL_PATTERN.search(text))


def has_time_digits(text):
    return bool(DATE_PATTERN.search(text) or TIME_OF_DAY_PATTERN.search(text))


def detect_boxes(pan, rgb):
    """Preserve aspect ratio, pad to stride 32, and map to original pixels."""
    import cv2
    import numpy as np
    import torch
    from .models.pan import decode_clip, sort_box
    height, width = rgb.shape[:2]
    scale = min(736 / min(height, width), 1536 / max(height, width))
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = (-resized_h) % 32, (-resized_w) % 32
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float()[None].to(pan.device) / 255
    with torch.inference_mode():
        predictions = pan.net(tensor)[0]
        # The notebook threshold drops faint/small text on narrow receipts.
        _, boxes = decode_clip(predictions, threshold=.35)
    mapped = []
    for box in boxes:
        points = np.asarray(box, dtype=np.float32)
        if points[:, 0].mean() >= resized_w or points[:, 1].mean() >= resized_h:
            continue
        points *= np.array([width / resized_w, height / resized_h], dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        mapped.append(points)
    return sort_box(mapped) if mapped else []


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn").replace("đ", "d")


def clean_for_bert(text):
    # Same number cleaning and punctuation preparation as notebook cell 66.
    from string import punctuation
    text = re.sub(r"(\w)(\()(\w)", r"\1 \2\3", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"(\w)([),.:;]+)(\w)", r"\1\2 \3", text)
    text = re.sub(r"(\w)(\.\()(\w)", r"\1. (\3", text)
    return " ".join(t.strip(punctuation) for t in text.lower().split() if t.strip(punctuation))


def compute_field_confidence(chosen_regions, field_value):
    """Calculate field confidence in percentage (0.0% - 100.0%). Returns 0.0% if unextracted."""
    if not chosen_regions or not field_value.strip():
        return 0.0
    scores = []
    for r in chosen_regions:
        ocr_c = float(r.get("ocr_confidence", 0.0))
        if r.get("rule", False):
            cls_c = 1.0
        else:
            cls_c = float(r.get("classification_confidence", 1.0))
        scores.append(ocr_c * cls_c)
    if not scores:
        return 0.0
    return round(max(0.0, min(100.0, (sum(scores) / len(scores)) * 100.0)), 1)


def compute_overall_confidence(fields):
    """Average of the 4 field confidences (unextracted fields contribute 0%)."""
    return round(sum(fields[lbl].get("confidence", 0.0) for lbl in LABELS) / len(LABELS), 1)


def summarize(regions):
    """Accurate, robust summary across all 4 fields with dictionary boost and spatial heuristics."""
    fields = {}

    # === 1. SELLER ===
    seller_candidates = []
    for r in regions:
        text = r["text"].strip()
        if not text or len(text) < 2:
            continue
        f = fold(text)
        if any(d in f for d in SELLER_DISQUALIFIERS) or total_keyword(text) or has_time_digits(text):
            continue
        dict_match = any(term in f for term in STORE_DICT)
        if dict_match or r["label"] == "SELLER":
            seller_candidates.append((r, dict_match))

    chosen_seller = []
    if seller_candidates:
        best_seller, is_dict = max(seller_candidates, key=lambda x: (
            x[1],  # Store dictionary match first
            x[0]["label"] == "SELLER",
            -sum(p[1] for p in x[0]["box"]) / 4,  # Higher on the receipt
            x[0]["classification_confidence"] * x[0]["ocr_confidence"]
        ))
        chosen_seller = [best_seller]

    seller_ids = {r["id"] for r in chosen_seller}
    seller_val = " ".join(r["text"] for r in chosen_seller)
    seller_conf = compute_field_confidence(chosen_seller, seller_val)
    fields["SELLER"] = {
        "name": FIELD_NAMES["SELLER"],
        "value": seller_val,
        "confidence": seller_conf,
        "confidence_display": f"{seller_conf:.1f}%" if seller_conf > 0 else "0%",
        "region_ids": list(seller_ids),
    }

    # === 2. TIMESTAMP ===
    time_candidates = []
    for r in regions:
        text = r["text"].strip()
        if not text:
            continue
        f = fold(text)
        if has_time_digits(text) and (r.get("rule", False) or r["label"] == "TIMESTAMP" or "ngay" in f or "gio" in f or "time" in f):
            if not any(k in f for k in ["tong", "thanh toan", "cam on", "doi, tra hang"]):
                time_candidates.append(r)

    chosen_time = []
    if time_candidates:
        best_time = max(time_candidates, key=lambda r: (
            bool(DATE_PATTERN.search(r["text"])) and bool(TIME_OF_DAY_PATTERN.search(r["text"])),
            bool(DATE_PATTERN.search(r["text"])),
            r.get("rule", False),
            r["classification_confidence"] * r["ocr_confidence"]
        ))
        chosen_time = [best_time]
        anchor_y = sum(p[1] for p in best_time["box"]) / 4
        anchor_h = max(p[1] for p in best_time["box"]) - min(p[1] for p in best_time["box"])
        for r in time_candidates:
            if r["id"] == best_time["id"]:
                continue
            cy = sum(p[1] for p in r["box"]) / 4
            if abs(cy - anchor_y) <= max(anchor_h * 1.5, 15):
                if r["text"] not in best_time["text"]:
                    chosen_time.append(r)
        chosen_time.sort(key=lambda r: (sum(p[1] for p in r["box"])/4, sum(p[0] for p in r["box"])/4))

    time_val = " ".join(r["text"] for r in chosen_time)
    time_ids = {r["id"] for r in chosen_time}
    time_conf = compute_field_confidence(chosen_time, time_val)
    fields["TIMESTAMP"] = {
        "name": FIELD_NAMES["TIMESTAMP"],
        "value": time_val,
        "confidence": time_conf,
        "confidence_display": f"{time_conf:.1f}%" if time_conf > 0 else "0%",
        "region_ids": list(time_ids),
    }

    # === 3. TOTAL_COST ===
    total_candidates = [
        r for r in regions
        if r["id"] not in seller_ids and r["id"] not in time_ids
        and total_keyword(r["text"]) and r["ocr_confidence"] >= 0.4
    ]

    chosen_total = []
    total_val = ""
    if total_candidates:
        best_total = max(total_candidates, key=lambda r: (
            AMOUNT_PATTERN.search(r["text"]) is not None,
            r["classification_confidence"] * r["ocr_confidence"]
        ))
        chosen_total = [best_total]
        total_val = best_total["text"]

        if not AMOUNT_PATTERN.search(total_val):
            left = max(p[0] for p in best_total["box"])
            cy = sum(p[1] for p in best_total["box"]) / 4
            height = max(p[1] for p in best_total["box"]) - min(p[1] for p in best_total["box"])
            neighbors = [
                r for r in regions if r["id"] != best_total["id"]
                and r["id"] not in seller_ids and r["id"] not in time_ids
                and (amount_only(r["text"]) or AMOUNT_PATTERN.search(r["text"]))
                and min(p[0] for p in r["box"]) >= left - height * 0.4
                and abs(sum(p[1] for p in r["box"]) / 4 - cy) <= max(height * 0.75, 10)
            ]
            if neighbors:
                neighbor = min(neighbors, key=lambda r: min(p[0] for p in r["box"]))
                chosen_total.append(neighbor)
                total_val += " " + neighbor["text"]

    total_ids = {r["id"] for r in chosen_total}
    total_conf = compute_field_confidence(chosen_total, total_val)
    fields["TOTAL_COST"] = {
        "name": FIELD_NAMES["TOTAL_COST"],
        "value": total_val,
        "confidence": total_conf,
        "confidence_display": f"{total_conf:.1f}%" if total_conf > 0 else "0%",
        "region_ids": list(total_ids),
    }

    # === 4. ADDRESS ===
    addr_candidates = []
    for r in regions:
        text = r["text"].strip()
        if not text or r["id"] in seller_ids or r["id"] in time_ids or r["id"] in total_ids:
            continue
        f = fold(text)
        if any(d in f for d in ADDRESS_DISQUALIFIERS):
            continue
        has_ind = any(ind in f for ind in ADDRESS_INDICATORS)
        if has_ind or (r["label"] == "ADDRESS" and r["classification_confidence"] >= 0.5):
            addr_candidates.append((r, has_ind))

    chosen_addr = []
    if addr_candidates:
        best_addr, has_ind = max(addr_candidates, key=lambda x: (
            x[1],
            x[0]["label"] == "ADDRESS",
            x[0]["classification_confidence"] * x[0]["ocr_confidence"]
        ))
        chosen_addr = [best_addr]
        ay = sum(p[1] for p in best_addr["box"]) / 4
        ah = max(p[1] for p in best_addr["box"]) - min(p[1] for p in best_addr["box"])
        for r, ind in addr_candidates:
            if r["id"] == best_addr["id"]:
                continue
            cy = sum(p[1] for p in r["box"]) / 4
            if abs(cy - ay) <= max(ah, 12) * 2.5:
                f_r = fold(r["text"])
                if not any(d in f_r for d in ADDRESS_DISQUALIFIERS):
                    chosen_addr.append(r)
        chosen_addr.sort(key=lambda r: (sum(p[1] for p in r["box"])/4, sum(p[0] for p in r["box"])/4))

    addr_val = " ".join(r["text"] for r in chosen_addr)
    addr_conf = compute_field_confidence(chosen_addr, addr_val)
    fields["ADDRESS"] = {
        "name": FIELD_NAMES["ADDRESS"],
        "value": addr_val,
        "confidence": addr_conf,
        "confidence_display": f"{addr_conf:.1f}%" if addr_conf > 0 else "0%",
        "region_ids": [r["id"] for r in chosen_addr],
    }

    return fields


class InvoiceEngine:
    def __init__(self):
        self.loaded = False
        self.device = "cpu"

    def load(self, progress):
        if self.loaded:
            return
        progress("loading", 5, "Đang nạp mô hình cho lần đọc đầu tiên…")
        import torch
        import yaml
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        from .models.pan import PAN
        from .models.ocr import build_model

        torch.set_num_threads(4)
        requested = os.getenv("OCR_DEVICE", "auto")
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        required = [ROOT / "weights" / n for n in ("PANNet_best_map.pth", "transformerocr.pth", "bert_weight.pth")]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Thiếu trọng số: {path.name}. Chuẩn bị model_assets trước khi chạy demo.")
        progress("loading", 7, "Đang nạp mô hình phát hiện vùng chữ…")
        pan_config = yaml.safe_load((ROOT / "configs/detection/configs.yaml").read_text(encoding="utf-8"))
        # Load checkpoint on CPU first; move only the completed model to the chosen device.
        state = torch.load(required[0], map_location="cpu", weights_only=True)
        self.pan = PAN.__new__(PAN)
        from .models.pan import get_model
        pan_config["model"]["pretrained"] = False
        self.pan.net = get_model(pan_config["model"])
        self.pan.net.load_state_dict(state["state_dict"])
        self.pan.device = torch.device(self.device)
        self.pan.net.to(self.device).eval()
        del state

        progress("loading", 9, "Đang nạp mô hình nhận dạng chữ tiếng Việt…")
        cfg_dir = ROOT / "configs/ocr/models"
        self.ocr_config = yaml.safe_load((cfg_dir / "base.yml").read_text(encoding="utf-8"))
        self.ocr_config.update(yaml.safe_load((cfg_dir / "vgg-transformer.yml").read_text(encoding="utf-8")))
        self.ocr_config["cnn"]["pretrained"] = False
        self.ocr_config["device"] = "cpu"
        self.ocr, self.vocab = build_model(self.ocr_config)
        self.ocr.load_state_dict(torch.load(required[1], map_location="cpu", weights_only=True))
        self.ocr.to(self.device).eval()
        progress("loading", 12, "Đang nạp mô hình phân loại trường thông tin…")
        checkpoint = torch.load(required[2], map_location="cpu", weights_only=True)
        args = checkpoint["config"]["model"]["args"]
        offline = os.getenv("OCR_ALLOW_DOWNLOAD", "0") != "1"
        config = AutoConfig.from_pretrained(args["pretrained_model"], num_labels=args["num_labels"], local_files_only=offline)
        self.bert = AutoModelForSequenceClassification.from_config(config)
        weights = {k.removeprefix("model."): v for k, v in checkpoint["model_state_dict"].items()}
        # Old Transformers checkpoints persist this derived, non-learned buffer.
        # New versions generate it internally; retain strict checking for all weights.
        if "roberta.embeddings.position_ids" not in self.bert.state_dict():
            weights.pop("roberta.embeddings.position_ids", None)
        self.bert.load_state_dict(weights)
        self.bert.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args["pretrained_model"], local_files_only=offline)
        del checkpoint, weights
        self.loaded = True

    def recognize_batch(self, images):
        import numpy as np
        import torch
        from .models.ocr import process_input
        cfg = self.ocr_config["dataset"]
        tensors = [process_input(im, cfg["image_height"], cfg["image_min_width"], cfg["image_max_width"]) for im in images]
        with torch.inference_mode():
            batch = torch.cat(tensors).to(self.device)
            memory = self.ocr.transformer.forward_encoder(self.ocr.cnn(batch))
            tokens = torch.ones((1, len(images)), dtype=torch.long, device=self.device)
            finished = torch.zeros(len(images), dtype=torch.bool, device=self.device)
            totals = torch.zeros(len(images), device=self.device)
            lengths = torch.zeros(len(images), device=self.device)
            for _ in range(129):
                logits, memory = self.ocr.transformer.forward_decoder(tokens, memory)
                probabilities, ids = logits[:, -1, :].softmax(-1).max(-1)
                valid = (~finished) & (ids > 3)
                totals += probabilities * valid
                lengths += valid
                finished |= ids.eq(2)
                tokens = torch.cat((tokens, ids[None, :]), dim=0)
                if bool(finished.all()):
                    break
            texts = self.vocab.batch_decode(tokens.T.cpu().tolist())
            confidence = (totals / lengths.clamp_min(1)).cpu().tolist()
        return list(zip(texts, confidence))

    def process(self, image_path, progress):
        started = time.monotonic()
        self.load(progress)
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from collections import defaultdict
        from .models.ocr import resize

        progress("detecting", 15, "Đang xác định các vùng chữ…")
        original = np.array(Image.open(image_path).convert("RGB"))
        height, width = original.shape[:2]
        boxes = detect_boxes(self.pan, original)
        crops, regions = [], []
        cfg = self.ocr_config["dataset"]
        groups = defaultdict(list)
        for box in boxes:
            box = np.asarray(box, dtype=np.float32)
            w = int(np.linalg.norm(box[0] - box[1]))
            h = int(np.linalg.norm(box[0] - box[3]))
            if min(w, h) < 2:
                continue
            dest = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            crop = cv2.warpPerspective(original, cv2.getPerspectiveTransform(box, dest), (w, h), borderMode=cv2.BORDER_REPLICATE)
            image = Image.fromarray(crop)
            index = len(crops)
            crops.append(image)
            groups[resize(w, h, cfg["image_height"], cfg["image_min_width"], cfg["image_max_width"])[0]].append(index)
            regions.append({"id": index, "box": box.round(2).tolist(), "text": "", "label": "NONE", "ocr_confidence": 0.0, "classification_confidence": 0.0, "rule": False})
        progress("reading", 25, f"Đã tìm thấy {len(crops)} vùng chữ. Đang nhận dạng…")
        done = 0
        batch_size = max(1, min(16, int(os.getenv("OCR_BATCH_SIZE", "8"))))
        for indices in groups.values():
            for offset in range(0, len(indices), batch_size):
                subset = indices[offset:offset + batch_size]
                predictions = self.recognize_batch([crops[i] for i in subset])
                for index, (text, score) in zip(subset, predictions):
                    regions[index].update(text=text, ocr_confidence=round(float(score), 4))
                done += len(subset)
                progress("reading", 25 + round(50 * done / max(1, len(crops))), f"Đã đọc {done}/{len(crops)} vùng chữ.")
        progress("classifying", 80, "Đang xác định tên cửa hàng, địa chỉ, ngày giờ và tổng tiền…")
        nonempty = [r for r in regions if r["text"].strip()]
        with torch.inference_mode():
            for offset in range(0, len(nonempty), 16):
                items = nonempty[offset:offset + 16]
                inputs = self.tokenizer([clean_for_bert(r["text"]) for r in items], padding="max_length", truncation=True, max_length=31, return_tensors="pt").to(self.device)
                probs, ids = self.bert(**inputs).logits.softmax(-1).max(-1)
                for region, label, probability in zip(items, ids.cpu().tolist(), probs.cpu().tolist()):
                    text = region["text"]
                    predicted = LABELS[label]
                    rule = False
                    if TIME_PATTERN.search(text):
                        predicted, rule = "TIMESTAMP", True
                    elif total_keyword(text):
                        predicted, rule = "TOTAL_COST", True
                    region.update(label=predicted, classification_confidence=round(probability, 4), rule=rule)
                progress("classifying", 80 + round(15 * min(offset + 16, len(nonempty)) / max(1, len(nonempty))), "Đang ghép các trường thông tin…", regions=regions)
        fields = summarize(regions)
        overall_conf = compute_overall_confidence(fields)
        return {
            "pipeline_version": "aspect-ratio-original-crops-v2",
            "width": width, "height": height, "regions": regions,
            "fields": fields,
            "overall_confidence": overall_conf,
            "overall_confidence_display": f"{overall_conf:.1f}%",
            "device": self.device,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "region_count": len(regions),
            "notice": "Không tìm thấy vùng chữ. Hãy thử ảnh rõ hơn, chụp gần và đủ sáng." if not regions else "",
        }
