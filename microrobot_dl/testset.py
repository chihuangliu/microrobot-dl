import json
import random
from pathlib import Path
import logging
from importlib.resources import files
from .config import TEST_RATIO

TEST_SET_PATH = files("microrobot_dl").joinpath("test_set_imagedataset2025.json")
SEED = 60648


def generate_imagedataset2025_test_set(
    imagedataset2025,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
    output_path: str = TEST_SET_PATH,
):
    """
    Generate a reproducible test split (list of indices) for an ImageDataset2025 instance
    and save the indices (plus minimal metadata) to `output_path` as JSON.

    Returns:
        list[int]: sorted list of test indices.
    """
    total = len(imagedataset2025)
    if total == 0:
        raise ValueError("Provided dataset is empty")

    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError("test_ratio must be in [0.0, 1.0]")

    # Determine test set size; if ratio > 0 but computes to 0 due to rounding, pick at least 1
    test_size = int(round(test_ratio * total))
    if test_ratio > 0 and test_size == 0:
        test_size = 1
    if test_size > total:
        test_size = total

    rng = random.Random(seed)
    if test_size >= total:
        indices = list(range(total))
    else:
        indices = rng.sample(range(total), test_size)

    indices_sorted = sorted(indices)

    output = {
        "test_indices": indices_sorted,
        "seed": seed,
        "test_ratio": test_ratio,
        "total_samples": total,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write test indices to {out_path}: {e}")
        raise

    return indices_sorted


def get_imagedataset2025_test_set(
    input_path: str = TEST_SET_PATH,
) -> dict:
    """
    Load the test set indices and metadata from a JSON file.

    Returns:
        dict: Dictionary containing test indices and metadata.
    """
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Test set file not found: {in_path}")

    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read test indices from {in_path}: {e}")
        raise

    required_keys = {"test_indices", "seed", "test_ratio", "total_samples"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(
            f"Test set file is missing required keys: {required_keys - data.keys()}"
        )

    return data


if __name__ == "__main__":
    from microrobot_dl.data_loader import ImageDataset2025

    dataset = ImageDataset2025(mode="pose")
    test_indices = generate_imagedataset2025_test_set(imagedataset2025=dataset)
    print(f"Generated test set with {len(test_indices)} indices.")
