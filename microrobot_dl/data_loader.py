import re
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import logging
from .config import VAL_RATIO, TEST_RATIO


class ImageDataset2025(Dataset):
    sub_dir_pattern = re.compile(r"^(P\d+_R\d+)$", re.IGNORECASE)
    split_pattern = re.compile(r"^P(\d+)_R(\d+)$", re.IGNORECASE)

    def __init__(
        self,
        base_dir: Path = "data/2025_Dataset",
        transform=None,
        mode=None,
        multi_label: bool = False,
        multi_task: bool = False,
    ):
        """
        Args:
            base_dir: Base directory containing the dataset.
            transform: Optional transform to be applied on images
            mode: 'depth' or 'pose'
            multi_label: If True, treats P and R as separate labels (returns [P, R]).
            multi_task: If True, returns (image, (pose_label, depth_label)).
        """
        if mode not in ["depth", "pose"] and not multi_task:
            raise ValueError(
                "label must be either 'depth' or 'pose' or multi_task=True"
            )
        self.mode = mode
        self.multi_label = multi_label
        self.multi_task = multi_task
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.samples = []
        self.label_to_idx = {}
        self.idx_to_label = []

        self.label_to_idx_p = {}
        self.idx_to_label_p = []
        self.label_to_idx_r = {}
        self.idx_to_label_r = []

        if self.multi_task:
            self._load_multi_task_data()
        elif self.mode == "depth":
            self._load_depth_data()
        elif self.mode == "pose":
            self._load_pose_data()

    def _load_multi_task_data(self):
        if not self.base_dir.exists():
            logging.warning(f"Base directory {self.base_dir} does not exist.")
            return

        for subdir in sorted(self.base_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not self.sub_dir_pattern.match(subdir.name):
                continue

            # 1. Parse Pose Label
            label_str = subdir.name
            if label_str not in self.label_to_idx:
                self.label_to_idx[label_str] = len(self.idx_to_label)
                self.idx_to_label.append(label_str)

            pose_label_data = None
            if self.multi_label:
                match = self.split_pattern.match(subdir.name)
                if match:
                    p_val = int(match.group(1))
                    r_val = int(match.group(2))

                    if p_val not in self.label_to_idx_p:
                        self.label_to_idx_p[p_val] = len(self.idx_to_label_p)
                        self.idx_to_label_p.append(p_val)

                    if r_val not in self.label_to_idx_r:
                        self.label_to_idx_r[r_val] = len(self.idx_to_label_r)
                        self.idx_to_label_r.append(r_val)

                    pose_label_data = (
                        self.label_to_idx_p[p_val],
                        self.label_to_idx_r[r_val],
                    )
            else:
                pose_label_data = self.label_to_idx[label_str]

            if pose_label_data is None and self.multi_label:
                continue

            # 2. Parse Depth Data
            txt_filename = f"{subdir.name}_depth.txt"
            txt_path = subdir / txt_filename
            depth_map = {}

            if txt_path.exists():
                with open(txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        content = line.strip("()")
                        parts = content.split(",")
                        if len(parts) != 2:
                            continue
                        img_name = parts[0].strip().strip("'").strip('"')
                        try:
                            depth_val = float(parts[1].strip())
                            depth_map[img_name] = depth_val
                        except ValueError:
                            continue

            # 3. Match Images
            jpg_files = sorted(subdir.rglob("*.jpg"))
            for jpg in jpg_files:
                if jpg.name in depth_map:
                    self.samples.append((jpg, pose_label_data, depth_map[jpg.name]))

        logging.info(
            f"Loaded {len(self.samples)} multi-task samples from {self.base_dir}"
        )

    def _load_depth_data(self):
        if not self.base_dir.exists():
            logging.warning(f"Base directory {self.base_dir} does not exist.")
            return

        for subdir in sorted(self.base_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not self.sub_dir_pattern.match(subdir.name):
                continue

            # Expected text file name: {dir_name}_depth.txt
            txt_filename = f"{subdir.name}_depth.txt"
            txt_path = subdir / txt_filename

            if not txt_path.exists():
                logging.warning(f"Depth label file not found: {txt_path}")
                continue

            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Format: ('filename', depth)
                    # Remove parentheses
                    content = line.strip("()")
                    parts = content.split(",")

                    if len(parts) != 2:
                        logging.warning(f"Invalid line format in {txt_path}: {line}")
                        continue

                    # Clean up filename (remove quotes and whitespace)
                    img_name = parts[0].strip().strip("'").strip('"')
                    try:
                        depth_val = float(parts[1].strip())
                    except ValueError:
                        logging.warning(f"Invalid depth value in {txt_path}: {line}")
                        continue

                    img_path = subdir / img_name
                    self.samples.append((img_path, depth_val))

        logging.info(f"Loaded {len(self.samples)} depth samples from {self.base_dir}")

    def _load_pose_data(self):
        if not self.base_dir.exists():
            logging.warning(f"Base directory {self.base_dir} does not exist.")
            return

        for subdir in sorted(self.base_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not self.sub_dir_pattern.match(subdir.name):
                continue

            label_str = subdir.name
            # Create label mapping (populate for both multi_label and single label modes)
            if label_str not in self.label_to_idx:
                self.label_to_idx[label_str] = len(self.idx_to_label)
                self.idx_to_label.append(label_str)

            if self.multi_label:
                match = self.split_pattern.match(subdir.name)
                if match:
                    p_val = int(match.group(1))
                    r_val = int(match.group(2))

                    if p_val not in self.label_to_idx_p:
                        self.label_to_idx_p[p_val] = len(self.idx_to_label_p)
                        self.idx_to_label_p.append(p_val)

                    if r_val not in self.label_to_idx_r:
                        self.label_to_idx_r[r_val] = len(self.idx_to_label_r)
                        self.idx_to_label_r.append(r_val)

                    label_data = (
                        self.label_to_idx_p[p_val],
                        self.label_to_idx_r[r_val],
                    )
                else:
                    logging.warning(f"Could not parse P/R from {subdir.name}")
                    continue
            else:
                label_data = self.label_to_idx[label_str]

            jpg_files = sorted(subdir.rglob("*.jpg"))
            for jpg in jpg_files:
                self.samples.append((jpg, label_data))

        logging.info(
            f"Loaded {len(self.samples)} pose samples from {self.base_dir} with {len(self.idx_to_label)} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.multi_task:
            img_path, pose_label, depth_val = self.samples[idx]

            try:
                image = Image.open(img_path).convert("L")
            except FileNotFoundError:
                logging.warning(f"Image not found: {img_path}")
                raise

            if self.transform:
                image = self.transform(image)

            pose_target = torch.tensor(pose_label, dtype=torch.long)
            depth_target = torch.tensor(depth_val, dtype=torch.float32)

            return image, (pose_target, depth_target)

        if self.mode == "depth":
            img_path, depth_val = self.samples[idx]

            try:
                image = Image.open(img_path).convert("L")
                label = torch.tensor(depth_val, dtype=torch.float32)
            except FileNotFoundError:
                logging.warning(f"Image not found: {img_path}")
                # Return a dummy or handle error appropriately;
                # here we might crash or return None, but standard is to assume data exists
                raise

            if self.transform:
                image = self.transform(image)

            return image, label

        elif self.mode == "pose":
            img_path, label_data = self.samples[idx]

            try:
                image = Image.open(img_path).convert("L")
            except FileNotFoundError:
                logging.warning(f"Image not found: {img_path}")
                raise

            if self.transform:
                image = self.transform(image)

            if self.multi_label:
                # label_data is (p_idx, r_idx)
                return image, torch.tensor(label_data, dtype=torch.long)
            else:
                # label_data is class index
                return image, torch.tensor(label_data, dtype=torch.long)


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    val_ratio: float = VAL_RATIO / (1 - TEST_RATIO),
    train_batch_size: int = 16,
    test_batch_size: int = 16,
    val_batch_size: int = 16,
    transform_train=None,
    transform_test=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the provided train_dataset into train and validation sets using train_ratio and val_ratio.
    Uses the provided test_dataset as-is for the test DataLoader.

    Args:
        train_dataset: Dataset to be split into train/validation.
        test_dataset: Dataset to be used for testing (not split).
        val_ratio: Proportion of train_dataset to allocate to validation (0 < val_ratio < 1).
        transform_train: Transform to apply to the training set.
        transform_test: Transform to apply to the validation and test sets.
    Returns:
        (train_loader, test_loader, val_loader)
    """

    total = len(train_dataset)
    if total == 0:
        raise ValueError("train_dataset is empty")

    train_size = int((1 - val_ratio) * total)
    val_size = int(val_ratio * total)

    # Assign any rounding remainder to the train set to ensure total sizes sum to dataset length
    remainder = total - (train_size + val_size)
    train_size += remainder

    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            "Resulting train or val split has zero samples; adjust ratios or provide more data"
        )

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    if transform_train:
        train_set = TransformedDataset(train_set, transform_train)

    if transform_test:
        val_set = TransformedDataset(val_set, transform_test)
        test_dataset = TransformedDataset(test_dataset, transform_test)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
