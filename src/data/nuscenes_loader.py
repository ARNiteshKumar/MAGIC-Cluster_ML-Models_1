"""
nuScenes Dataset Loader for Simple-BEV Validation
===================================================
Loads the real nuScenes dataset (mini / trainval / test splits) and
produces (multi_cam_images, bev_segmentation_labels) pairs compatible
with the SimpleBEV model.

Camera order: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
              CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

BEV ground truth is built by projecting 3D bounding-box annotations
onto a top-down 200x200 grid that covers [-50m, 50m] x [-50m, 50m]
around the ego vehicle.

Usage (standalone test):
    python src/data/nuscenes_loader.py --data_root /path/to/nuscenes --version v1.0-mini
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# nuScenes category -> BEV class ID mapping
# ---------------------------------------------------------------------------
#   0: background
#   1: drivable_surface  (from map / fallback: empty road)
#   2: vehicle           (car, truck, bus, trailer, construction_vehicle)
#   3: pedestrian
#   4: cyclist           (bicycle, motorcycle)
#   5: road_marking      (not available from boxes; left as background)
#   6: static_obstacle   (barrier, traffic_cone, debris)
#   7: other
# ---------------------------------------------------------------------------
CATEGORY_TO_CLASS = {
    "human.pedestrian.adult": 3,
    "human.pedestrian.child": 3,
    "human.pedestrian.wheelchair": 3,
    "human.pedestrian.stroller": 3,
    "human.pedestrian.personal_mobility": 3,
    "human.pedestrian.police_officer": 3,
    "human.pedestrian.construction_worker": 3,
    "vehicle.car": 2,
    "vehicle.motorcycle": 4,
    "vehicle.bicycle": 4,
    "vehicle.bus.bendy": 2,
    "vehicle.bus.rigid": 2,
    "vehicle.truck": 2,
    "vehicle.trailer": 2,
    "vehicle.construction": 2,
    "vehicle.emergency.ambulance": 2,
    "vehicle.emergency.police": 2,
    "movable_object.barrier": 6,
    "movable_object.trafficcone": 6,
    "movable_object.pushable_pullable": 6,
    "movable_object.debris": 6,
    "static_object.bicycle_rack": 6,
}

CAMERA_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# BEV grid parameters (metres)
BEV_X_RANGE = (-50.0, 50.0)  # left-right
BEV_Y_RANGE = (-50.0, 50.0)  # front-back
BEV_RESOLUTION = 0.5         # metres per pixel  -> 200 x 200 grid


def _world_to_bev(x: float, y: float, bev_h: int = 200, bev_w: int = 200):
    """Convert world coordinates (ego frame) to BEV pixel coordinates."""
    px = int((x - BEV_X_RANGE[0]) / (BEV_X_RANGE[1] - BEV_X_RANGE[0]) * bev_w)
    py = int((y - BEV_Y_RANGE[0]) / (BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) * bev_h)
    px = np.clip(px, 0, bev_w - 1)
    py = np.clip(py, 0, bev_h - 1)
    return px, py


def _rotation_matrix(quaternion):
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    try:
        from pyquaternion import Quaternion
        q = Quaternion(quaternion)
        return q.rotation_matrix
    except ImportError:
        # manual fallback
        w, x, y, z = quaternion
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class NuScenesLoader:
    """
    Loads nuScenes samples and produces numpy arrays ready for inference.

    Parameters
    ----------
    data_root : str
        Path to the nuScenes dataset root (contains maps/, samples/, sweeps/, ...).
    version : str
        Dataset version, e.g. "v1.0-mini", "v1.0-trainval", "v1.0-test".
    split : str
        "train", "val", or "test".  For mini, use "mini_train" / "mini_val".
    img_h, img_w : int
        Target image size for the model (default 224 x 400).
    bev_h, bev_w : int
        BEV grid size (default 200 x 200).
    max_samples : int or None
        Cap the number of samples loaded (useful for quick testing).
    """

    def __init__(self, data_root: str, version: str = "v1.0-mini",
                 split: str = "mini_val",
                 img_h: int = 224, img_w: int = 400,
                 bev_h: int = 200, bev_w: int = 200,
                 max_samples: int = None):
        from nuscenes.nuscenes import NuScenes

        self.img_h = img_h
        self.img_w = img_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.data_root = data_root

        print(f"  Loading nuScenes {version} from {data_root} ...")
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

        # Collect scene tokens for the requested split
        self.samples = self._get_split_samples(split, max_samples)
        print(f"  Found {len(self.samples)} samples in split '{split}'")

    def _get_split_samples(self, split: str, max_samples):
        """Return list of sample records for the given split."""
        from nuscenes.utils.splits import create_splits_scenes

        split_scenes = create_splits_scenes()

        # Map common split names
        split_key = split
        if split_key not in split_scenes:
            # try mapping
            mapping = {
                "train": "train", "val": "val", "test": "test",
                "mini_train": "mini_train", "mini_val": "mini_val",
            }
            split_key = mapping.get(split, split)

        if split_key not in split_scenes:
            print(f"  WARNING: split '{split}' not found, using all scenes")
            scene_names = {s["name"] for s in self.nusc.scene}
        else:
            scene_names = set(split_scenes[split_key])

        scene_tokens = {
            s["token"] for s in self.nusc.scene if s["name"] in scene_names
        }

        samples = []
        for sample in self.nusc.sample:
            if sample["scene_token"] in scene_tokens:
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break
        return samples

    def __len__(self):
        return len(self.samples)

    def load_sample(self, idx: int):
        """
        Load a single nuScenes sample.

        Returns
        -------
        imgs : np.ndarray, shape (6, 3, img_h, img_w), float32, normalised [0,1]
        bev_label : np.ndarray, shape (bev_h, bev_w), int64
        sample_token : str
        """
        sample = self.samples[idx]
        imgs = self._load_cameras(sample)
        bev_label = self._build_bev_label(sample)
        return imgs, bev_label, sample["token"]

    def _load_cameras(self, sample) -> np.ndarray:
        """Load and preprocess 6 camera images, returning (6, 3, H, W)."""
        cam_imgs = []
        for cam in CAMERA_CHANNELS:
            cam_token = sample["data"][cam]
            cam_data = self.nusc.get("sample_data", cam_token)
            img_path = os.path.join(self.data_root, cam_data["filename"])

            img = cv2.imread(img_path)
            if img is None:
                # fallback: black image
                img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_w, self.img_h))

            # HWC -> CHW, normalise to [0, 1]
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # (3, H, W)
            cam_imgs.append(img)

        return np.stack(cam_imgs, axis=0)  # (6, 3, H, W)

    def _build_bev_label(self, sample) -> np.ndarray:
        """
        Build a BEV segmentation label from 3D annotation boxes.

        Each annotation box is projected onto the BEV plane as a filled
        rotated rectangle with the class ID from CATEGORY_TO_CLASS.

        Returns (bev_h, bev_w) int64 array.
        """
        bev = np.zeros((self.bev_h, self.bev_w), dtype=np.int64)

        # Get ego pose for this sample's lidar
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_rot = np.array(_rotation_matrix(ego_pose["rotation"]))
        ego_trans = np.array(ego_pose["translation"])

        # Drivable surface: fill entire grid as class 1 (basic baseline)
        # A proper implementation would use the nuScenes map API
        bev[:] = 1  # drivable_surface as default

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            category = ann["category_name"]
            cls_id = CATEGORY_TO_CLASS.get(category, 7)  # 7 = other

            # Get box center and size in global frame
            center_global = np.array(ann["translation"])  # (x, y, z)
            size = ann["size"]  # (w, l, h) in metres
            rot_global = _rotation_matrix(ann["rotation"])

            # Transform to ego frame
            center_ego = ego_rot.T @ (center_global - ego_trans)

            # Skip if outside BEV range
            if (center_ego[0] < BEV_X_RANGE[0] - 5 or
                center_ego[0] > BEV_X_RANGE[1] + 5 or
                center_ego[1] < BEV_Y_RANGE[0] - 5 or
                center_ego[1] > BEV_Y_RANGE[1] + 5):
                continue

            # Build 2D footprint corners (top-down: x, y)
            w, l = size[0], size[1]  # width, length
            corners_local = np.array([
                [-w / 2, -l / 2],
                [+w / 2, -l / 2],
                [+w / 2, +l / 2],
                [-w / 2, +l / 2],
            ])

            # Rotate corners by the annotation yaw in ego frame
            rot_ego = ego_rot.T @ rot_global
            yaw = np.arctan2(rot_ego[1, 0], rot_ego[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            R2d = np.array([[c, -s], [s, c]])
            corners_ego_2d = (R2d @ corners_local.T).T + center_ego[:2]

            # Convert to BEV pixel coords
            corners_bev = []
            for cx, cy in corners_ego_2d:
                px, py = _world_to_bev(cx, cy, self.bev_h, self.bev_w)
                corners_bev.append([px, py])
            corners_bev = np.array(corners_bev, dtype=np.int32)

            # Draw filled polygon on BEV grid
            cv2.fillPoly(bev, [corners_bev], int(cls_id))

        return bev

    def load_all(self):
        """
        Load all samples into numpy arrays.

        Returns
        -------
        all_imgs : (N, 6, 3, H, W) float32
        all_labels : (N, bev_h, bev_w) int64
        tokens : list of str
        """
        from tqdm import tqdm
        all_imgs, all_labels, tokens = [], [], []
        for idx in tqdm(range(len(self.samples)), desc="Loading nuScenes"):
            imgs, lbl, tok = self.load_sample(idx)
            all_imgs.append(imgs)
            all_labels.append(lbl)
            tokens.append(tok)
        return (np.stack(all_imgs).astype(np.float32),
                np.stack(all_labels).astype(np.int64),
                tokens)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test nuScenes loader")
    parser.add_argument("--data_root", required=True,
                        help="Path to nuScenes dataset root")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_val")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    loader = NuScenesLoader(
        data_root=args.data_root,
        version=args.version,
        split=args.split,
        max_samples=args.max_samples,
    )

    imgs, labels, tokens = loader.load_all()
    print(f"\nLoaded {len(tokens)} samples")
    print(f"  Images shape : {imgs.shape}")
    print(f"  Labels shape : {labels.shape}")
    print(f"  Label classes present: {np.unique(labels)}")

    # Print per-class pixel counts from first sample
    lbl = labels[0]
    print(f"\nSample 0 (token={tokens[0][:8]}...) class distribution:")
    for cls_id in range(8):
        count = (lbl == cls_id).sum()
        pct = count / lbl.size * 100
        name = ["background", "drivable_surface", "vehicle", "pedestrian",
                "cyclist", "road_marking", "static_obstacle", "other"][cls_id]
        print(f"  {cls_id} {name:<20s}: {count:6d} px ({pct:.1f}%)")


if __name__ == "__main__":
    main()
