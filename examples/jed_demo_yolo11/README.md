# Jed’s YOLOv11 Mini-Recipe (COCO128)

A reproducible <15-min sanity-check training & evaluation for **YOLOv11** on **COCO128**.

## Quickstart
```bash
conda env create -f env.yml && conda activate yolo11-jed
python train_detect.py --model yolov11n.pt --epochs 3 --imgsz 640
python eval_map.py --weights runs/detect/train/weights/best.pt --split val
```
