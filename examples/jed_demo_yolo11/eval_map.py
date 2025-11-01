import argparse
from ultralytics import YOLO

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val", choices=["val","test","train"])
    ap.add_argument("--device", default=None)
    return ap.parse_args()

def main():
    a = parse()
    model = YOLO(a.weights)
    metrics = model.val(imgsz=a.imgsz, split=a.split, device=a.device)
    print({k: float(metrics.results_dict.get(k, 0.0)) for k in [
        "metrics/mAP50","metrics/mAP50-95","metrics/precision","metrics/recall"
    ]})

if __name__ == "__main__":
    main()
