"""Regress narrow receipt and sample without touching the user's saved jobs."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from invoice_demo.engine import InvoiceEngine

engine = InvoiceEngine()
out = ROOT / 'output/fastapi_demo/alignment_review'
out.mkdir(exist_ok=True)
images = [ROOT / 'output/fastapi_demo/21e1d59e-d25c-440c-acb7-3707d4b52f3e/image.jpg', ROOT / 'data/val_images/val_images/mcocr_val_145114gpkup.jpg']
for index, path in enumerate(images):
    result = engine.process(path, lambda stage, percent, message, **extra: print(index, stage, percent, flush=True))
    (out / f'result-{index}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    canvas = cv2.imread(str(path))
    for r in result['regions']:
        cv2.polylines(canvas, [np.asarray(r['box'], dtype=np.int32)], True, (0,0,255), 1)
        print(r['id'], r['label'], r['text'], round(r['ocr_confidence'],2), flush=True)
    cv2.imwrite(str(out / f'boxes-{index}.jpg'), canvas)
    print('FIELDS', {key: value['value'] for key, value in result['fields'].items()}, flush=True)
