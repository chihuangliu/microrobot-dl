import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from microrobot_dl.task import Task
from microrobot_dl.model import (
    get_model,
    resnet_models,
    alexnet_models,
    vgg_models,
    vit_models,
)
from microrobot_dl.data_loader import ImageDataset2025
from microrobot_dl.testset import get_imagedataset2025_test_set
from microrobot_dl.inference import evaluate_model


def get_architecture_from_name(model_name_str):
    all_models = resnet_models + alexnet_models + vgg_models + vit_models
    all_models.sort(key=len, reverse=True)
    for m in all_models:
        if model_name_str.startswith(m):
            return m
    return None


def patch_metrics():
    eval_dir = "eval"
    model_dir = "model"
    data_dir = os.path.join("data", "2025_Dataset")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    tasks_to_process = [Task.pose_single, Task.pose_multi, Task.multi_tasks]
    dataset_cache = {}

    for task in tasks_to_process:
        task_dir = os.path.join(eval_dir, task)
        if not os.path.exists(task_dir):
            print(f"Task directory does not exist: {task_dir}")
            continue

        print(f"Processing task directory: {task}")
        files = [f for f in os.listdir(task_dir) if f.endswith(".json")]

        for filename in files:
            filepath = os.path.join(task_dir, filename)
            print(f"  Checking file: {filename}")

            with open(filepath, "r") as f:
                try:
                    content = json.load(f)
                except json.JSONDecodeError:
                    print(f"    Error decoding {filepath}")
                    continue

            data_list = content if isinstance(content, list) else [content]
            modified = False

            for i, entry in enumerate(data_list):
                model_name = entry.get("model_name")
                if not model_name:
                    continue

                model_path = os.path.join(model_dir, f"{model_name}.pth")
                if not os.path.exists(model_path):
                    print(f"    Model not found: {model_name}")
                    continue

                metadata = entry.get("metadata", {})
                is_multi_label = metadata.get("multi_label", False)

                if task == Task.pose_single:
                    ds_mode = "pose"
                    ds_multi_label = False
                elif task == Task.pose_multi:
                    ds_mode = "pose"
                    ds_multi_label = True
                elif task == Task.multi_tasks:
                    ds_mode = None
                    ds_multi_label = is_multi_label
                else:
                    continue

                ds_key = (ds_mode, ds_multi_label, task == Task.multi_tasks)

                if ds_key not in dataset_cache:
                    print(f"    Loading dataset for config: {ds_key}")
                    dataset = ImageDataset2025(
                        base_dir=data_dir,
                        mode=ds_mode,
                        multi_label=ds_multi_label,
                        multi_task=(task == Task.multi_tasks),
                        transform=None,
                    )
                    dataset_cache[ds_key] = dataset
                else:
                    dataset = dataset_cache[ds_key]

                num_classes_p = 0
                num_classes_r = 0
                if task == Task.pose_multi or (
                    task == Task.multi_tasks and ds_multi_label
                ):
                    num_classes_p = len(dataset.idx_to_label_p)
                    num_classes_r = len(dataset.idx_to_label_r)
                    num_outputs = num_classes_p + num_classes_r
                elif task == Task.pose_single or (
                    task == Task.multi_tasks and not ds_multi_label
                ):
                    num_outputs = len(dataset.idx_to_label)

                if task == Task.multi_tasks:
                    num_outputs += 1

                test_set_info = get_imagedataset2025_test_set()
                test_indices = list(test_set_info["test_indices"])
                test_subset = torch.utils.data.Subset(dataset, test_indices)

                transform_test = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                )
                dataset.transform = transform_test

                test_loader = torch.utils.data.DataLoader(
                    test_subset, batch_size=64, shuffle=False, num_workers=0
                )

                arch = get_architecture_from_name(model_name)
                if not arch:
                    print(f"    Unknown architecture for {model_name}")
                    continue

                model = get_model(arch, num_outputs=num_outputs, in_channels=1)
                model = model.to(device)

                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])

                    # Check learning rate
                    json_lr = metadata.get("learning_rate")
                    if json_lr is not None and "optimizer_state_dict" in checkpoint:
                        opt_state = checkpoint["optimizer_state_dict"]
                        if (
                            "param_groups" in opt_state
                            and len(opt_state["param_groups"]) > 0
                        ):
                            checkpoint_lr = opt_state["param_groups"][0].get("lr")
                            if (
                                checkpoint_lr is not None
                                and abs(checkpoint_lr - json_lr) > 1e-6
                            ):
                                print(
                                    f"      LR mismatch for {model_name}: Checkpoint {checkpoint_lr}, JSON {json_lr}"
                                )
                                continue

                except Exception as e:
                    print(f"    Failed to load model {model_name}: {e}")
                    continue

                criterion = None
                criterion_pose = None
                criterion_depth = None

                if task in [Task.pose_single, Task.pose_multi]:
                    criterion = nn.CrossEntropyLoss()
                elif task == Task.multi_tasks:
                    criterion_pose = nn.CrossEntropyLoss()
                    criterion_depth = nn.MSELoss()

                reg_weight = metadata.get("multi_tasks_regression_loss_weight", 8)

                print(f"    Evaluating {model_name}...")
                results = evaluate_model(
                    model,
                    test_loader,
                    device,
                    task,
                    criterion=criterion,
                    criterion_pose=criterion_pose,
                    criterion_depth=criterion_depth,
                    num_classes_p=num_classes_p,
                    num_classes_r=num_classes_r,
                    multi_label=ds_multi_label,
                    multi_tasks_regression_loss_weight=reg_weight,
                )

                calc_metric = results.get("accuracy", results.get("rmse", 0.0))
                stored_metric = entry.get("metric_value", 0.0)

                if abs(calc_metric - stored_metric) < 1e-4:
                    print("      Metrics match. Patching metadata.")
                    modified = True

                    if task == Task.pose_single or (task == Task.multi_tasks):
                        metadata["precision"] = results.get("precision", 0.0)
                        metadata["recall"] = results.get("recall", 0.0)
                        metadata["f1"] = results.get("f1", 0.0)

                    if task == Task.pose_multi or (
                        task == Task.multi_tasks and ds_multi_label
                    ):
                        metadata["precision_p"] = results.get("precision_p", 0.0)
                        metadata["recall_p"] = results.get("recall_p", 0.0)
                        metadata["f1_p"] = results.get("f1_p", 0.0)
                        metadata["precision_r"] = results.get("precision_r", 0.0)
                        metadata["recall_r"] = results.get("recall_r", 0.0)
                        metadata["f1_r"] = results.get("f1_r", 0.0)

                    entry["metadata"] = metadata
                    data_list[i] = entry
                else:
                    print(
                        f"      Metrics mismatch: Stored {stored_metric:.4f}, Calc {calc_metric:.4f}"
                    )

            if modified:
                with open(filepath, "w") as f:
                    json.dump(data_list, f, indent=2)
                print(f"    Saved updates to {filename}")


if __name__ == "__main__":
    patch_metrics()
