"""Export model definitions, without running notebook cells or demo side effects."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = json.loads((ROOT / "invoice-ocr-with-metrics.ipynb").read_text(encoding="utf-8"))["cells"]
TARGET = ROOT / "invoice_demo" / "models"


def definitions(indices, *, assignments=False, exclude=()):
    result = []
    for index in indices:
        tree = ast.parse("".join(CELLS[index]["source"]))
        for node in tree.body:
            allowed = isinstance(node, (ast.FunctionDef, ast.ClassDef))
            if assignments and isinstance(node, ast.Assign):
                allowed = True
            if allowed and getattr(node, "name", None) not in exclude:
                result.append(ast.unparse(node))
    return "\n\n".join(result) + "\n"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "__init__.py").write_text("", encoding="utf-8")
    notice = '"""Model definitions extracted from invoice-ocr-with-metrics.ipynb.\nRegenerate with: python tools/export_demo_models.py\nPAN and OCR live in separate modules to avoid notebook name collisions.\n"""\n'
    pan_imports = "import os, time\nimport cv2\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nimport torchvision.transforms as tf\nfrom torch.hub import load_state_dict_from_url\n"
    ocr_imports = "import math\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torchvision import models\nfrom torch.nn.functional import softmax, log_softmax\nfrom PIL import Image\n"
    (TARGET / "pan.py").write_text(notice + pan_imports + definitions([18, 21, 23, 27], assignments=True), encoding="utf-8")
    (TARGET / "ocr.py").write_text(notice + ocr_imports + definitions([40, 42, 44, 46, 48, 50, 52, 54], exclude={"predict", "batch_translate_beam_search"}), encoding="utf-8")
    print("Exported isolated PAN and VietOCR modules.")


if __name__ == "__main__":
    main()
