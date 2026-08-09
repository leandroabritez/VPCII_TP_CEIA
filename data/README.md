# Dataset Instructions

The dataset is hosted publicly on Roboflow Universe. It is the same dataset available on Kaggle.

## Download via Roboflow API (recommended)

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("veda").project("bone-fracture-detection-daoon")
version = project.version(1)
dataset = version.download("yolov11")
```

Place the downloaded folder inside `data/` so the structure is:
```
data/
└── bone-fracture-detection-daoon-1/
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml
```

## References

- Roboflow: https://universe.roboflow.com/veda/bone-fracture-detection-daoon
- Kaggle (same dataset): https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/data
