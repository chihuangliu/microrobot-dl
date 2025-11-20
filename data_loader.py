import re
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import logging


class ImageDepthDataset(Dataset):
    def __init__(self, data_dirs, base_dir: str = "data/Image_Depth", transform=None):
        """
        Args:
            data_dirs: List of directory names to load data from
            base_dir: Base directory path
            transform: Optional transform to be applied on images
        """
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.samples = []

        for dir_name in data_dirs:
            dir_path = self.base_dir.joinpath(dir_name)

            txt_files = dir_path.rglob("*.txt")

            _c = 0
            for _, txt_file in enumerate(txt_files):
                if txt_file.is_relative_to(dir_path.joinpath(".ipynb_checkpoints")):
                    continue

                if _c == 1:
                    logging.log(
                        logging.WARNING,
                        f"More than 1 text file found in {dir_path}, use the first one found.",
                    )
                    break

                labels = []
                label_filenames = []
                with open(txt_file, "r") as f:
                    print(f"reading {txt_file}")
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        # lines may be either a single float or like: "<filename> <id> <float>"
                        token = s.split()[-1]
                        filename = s.split()[0]
                        try:
                            labels.append(float(token))
                            label_filenames.append(filename)
                        except ValueError:
                            logging.warning(
                                f"Could not parse float from line in {txt_file}: {line!r}"
                            )
                            continue

                _c += 1

            for label, label_filename in zip(labels, label_filenames):
                self.samples.append((dir_path.joinpath(label_filename), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("L")
            label = torch.tensor(_label, dtype=torch.float32)
        except FileNotFoundError:
            logging.warning(f"{image} not found.")

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, label


class PoseImageDataset(Dataset):
    """Dataset that reads images from subfolders of base_dir/data_dir(s).
    Each matching subfolder (e.g., 'P0_R0', 'P20_R30') is a label;
    every .jpg inside that folder gets that label.
    Returns image tensor (C=1) and integer label index."""

    def __init__(
        self, data_dirs: list[str], base_dir: str = "data/Image_Pose", transform=None
    ):
        # data_dirs: list of dataset folders under base_dir, e.g. ['robot_1_four_ball', ...]
        self.base_dir = Path(base_dir)
        self.data_dirs = data_dirs
        self.transform = transform
        self.samples = []  # list of (image_path, label_idx)
        self.label_to_idx = {}  # label string -> int
        self.idx_to_label = []  # index -> label string

        for dir_name in self.data_dirs:
            dir_root = self.base_dir.joinpath(dir_name)
            if not dir_root.exists():
                logging.warning(f"Data directory does not exist: {dir_root}; skipping")
                continue

            # iterate subfolders of this dir_root and pick those like 'P..._R...' (case-insensitive 'P' prefix and containing '_R')
            for sub in sorted(dir_root.iterdir()):
                if not sub.is_dir():
                    continue
                name = sub.name
                if not (
                    (name.startswith("P") or name.startswith("p"))
                    and ("_R" in name or "_r" in name)
                ):
                    continue

                jpg_files = sorted(sub.rglob("*.jpg"))
                if not jpg_files:
                    logging.warning(f"No .jpg files in folder {sub}; skipping")
                    continue

                # global label mapping across all data_dirs
                if name not in self.label_to_idx:
                    self.label_to_idx[name] = len(self.idx_to_label)
                    self.idx_to_label.append(name)
                lbl_idx = self.label_to_idx[name]

                for jf in jpg_files:
                    self.samples.append((jf, lbl_idx))

        logging.info(
            f"Loaded {len(self.samples)} samples from {self.base_dir} ({len(self.idx_to_label)} labels)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_idx = self.samples[idx]
        img = Image.open(img_path).convert("L")  # single-channel
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(lbl_idx, dtype=torch.long)


class ImageDataset2025(Dataset):
    sub_dir_pattern = re.compile(r"^(P\d+_R\d+)$", re.IGNORECASE)
    split_pattern = re.compile(r"^P(\d+)_R(\d+)$", re.IGNORECASE)

    def __init__(
        self,
        base_dir: str = "data/2025_Dataset",
        transform=None,
        mode=None,
        multi_label: bool = False,
    ):
        """
        Args:
            data_dirs: List of directory names to load data from
            transform: Optional transform to be applied on images
            mode: 'depth' or 'pose'
            multi_label: If True, treats P and R as separate labels (returns [P, R]).
        """
        if mode not in ["depth", "pose"]:
            raise ValueError("label must be either 'depth' or 'pose'")
        self.mode = mode
        self.multi_label = multi_label
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.samples = []
        self.label_to_idx = {}
        self.idx_to_label = []

        self.label_to_idx_p = {}
        self.idx_to_label_p = []
        self.label_to_idx_r = {}
        self.idx_to_label_r = []

        if self.mode == "depth":
            self._load_depth_data()
        elif self.mode == "pose":
            self._load_pose_data()

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
