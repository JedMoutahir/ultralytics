import argparse
from ultralytics import YOLO

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov11n.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    return ap.parse_args()

def main():
    a = parse()
    model = YOLO(a.model)
    model.train(data="coco128.yaml", epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device)

if __name__ == "__main__":
    main()
