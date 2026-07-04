import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg

def crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

def ben_graham_preprocess(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4, blurred, -4, 128)
    return out

def preprocess_fundus(img: np.ndarray) -> np.ndarray:
    img = crop_black_borders(img)
    img = cv2.resize(img, (cfg.image_size, cfg.image_size))
    img = ben_graham_preprocess(img)
    return img

def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
        ToTensorV2(),
    ])

def get_eval_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
        ToTensorV2(),
    ])

class RetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: A.Compose):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = cfg.images_dir / f"{row['id_code']}.png"
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_fundus(img)
        img = self.transforms(image=img)["image"]
        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label