# Data Card — Simple-BEV

## Dataset Overview
| Field | Detail |
|-------|--------|
| Name | nuScenes (target) / Synthetic (this repo) |
| Task | Multi-camera BEV Semantic Segmentation |
| Input | 6 surround-view camera images per sample |
| Labels | 8-class BEV segmentation map |
| Image Size | 224 × 400 (H × W) |
| BEV Grid | 200 × 200 pixels |

## Synthetic Sample (Used in This Repo)
Since nuScenes requires registration, this repo uses synthetic data for pipeline validation.

```python
imgs   = torch.randn(B, 6, 3, 224, 400)
labels = torch.randint(0, 8, (B, 200, 200))
```

## Real Dataset: nuScenes
1. Register at https://www.nuscenes.org/
2. Download mini split (~4GB)
3. Update `configs/config.yaml` data_dir

## Class Map (8 Classes)
| 0 | Background | | 1 | Drivable surface | | 2 | Vehicle |
| 3 | Pedestrian | | 4 | Cyclist | | 5 | Road marking |
| 6 | Static obstacle | | 7 | Other |
