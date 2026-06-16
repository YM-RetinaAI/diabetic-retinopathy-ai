from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    data_dir: Path = Path("data/aptos2019")
    train_csv: Path = Path("data/aptos2019/train.csv")
    images_dir: Path = Path("data/aptos2019/train_images")
    artifacts_dir: Path = Path("ml/artifacts")
    image_size: int = 300
    num_classes: int = 5
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    

    def __post_init__(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

cfg = Config()