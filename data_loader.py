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
