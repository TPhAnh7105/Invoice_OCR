"""Compare detector preprocessing on the failing local receipt."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import yaml
from invoice_demo.models.pan import get_model, decode_clip

torch.set_num_threads(4)
config = yaml.safe_load((ROOT / 'configs/detection/configs.yaml').read_text())['model']
config['pretrained'] = False
net = get_model(config)
net.load_state_dict(torch.load(ROOT / 'weights/PANNet_best_map.pth', map_location='cpu', weights_only=True)['state_dict'])
net.eval()
source = cv2.imread(str(ROOT / 'output/fastapi_demo/21e1d59e-d25c-440c-acb7-3707d4b52f3e/image.jpg'))
h, w = source.shape[:2]
out = ROOT / 'output/fastapi_demo/detection_review'
out.mkdir(exist_ok=True)
for short_size in (736,):
    for threshold in (.5, .35):
        rgb, normalize = True, False
        image = cv2.resize(source, (short_size, round(h * short_size / w)))
        if rgb:
            image = image[:, :, ::-1].copy()
        tensor = torch.tensor(image.transpose(2,0,1).copy()).float()[None] / 255
        if normalize:
            tensor = (tensor - torch.tensor([.485,.456,.406])[None,:,None,None]) / torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            pred = net(tensor)[0]
            pred[1] = pred[0]
            _, boxes = decode_clip(pred, threshold=threshold)
        canvas = source.copy()
        scale = np.array([w / short_size, h / image.shape[0]])
        mapped = [np.asarray(box)*scale for box in boxes]
        for box in mapped:
            cv2.polylines(canvas, [box.astype(np.int32)], True, (0,0,255), 1)
        name = f'text_{short_size}_{threshold}'
        cv2.imwrite(str(out / (name+'.jpg')),canvas)
        print(name, len(boxes), flush=True)
