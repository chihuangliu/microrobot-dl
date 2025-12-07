import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Generator, Tuple, Any
from microrobot_dl.task import Task
from sklearn.metrics import precision_recall_fscore_support


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: Task,
    training: bool = False,
) -> Generator[Tuple[torch.Tensor, Any], None, None]:
    """
    Runs inference on the dataloader and yields (outputs, labels) for each batch.
    """
    if training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(training):
        for images, labels in dataloader:
            images = images.to(device)

            if task == Task.multi_tasks:
                pose_labels, depth_labels = labels
                pose_labels = pose_labels.to(device)
                depth_labels = depth_labels.to(device).float().unsqueeze(1)
                labels = (pose_labels, depth_labels)
            else:
                labels = labels.to(device)
                if task == Task.depth:
                    labels = labels.float().unsqueeze(1)

            outputs = model(images)
            yield outputs, labels


def evaluate_metrics(
    inference_results: Generator[Tuple[torch.Tensor, Any], None, None],
    task: str,
    dataset_len: int,
    criterion: Optional[nn.Module] = None,
    criterion_pose: Optional[nn.Module] = None,
    criterion_depth: Optional[nn.Module] = None,
    num_classes_p: int = 0,
    num_classes_r: int = 0,
    multi_label: bool = False,
    multi_tasks_regression_loss_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Calculates evaluation metrics from inference results.

    Args:
        inference_results (Generator[Tuple[torch.Tensor, Any], None, None]): A generator yielding
            batches of (outputs, labels) from the model inference.
        task (str): The task identifier (e.g., Task.pose_single, Task.multi_tasks).
        dataset_len (int): The total number of samples in the dataset, used for averaging loss.
        criterion (Optional[nn.Module], optional): Loss function for single-task scenarios
            (pose_single, pose_multi, depth). Defaults to None.
        criterion_pose (Optional[nn.Module], optional): Loss function for the pose component
            in multi-task scenarios. Defaults to None.
        criterion_depth (Optional[nn.Module], optional): Loss function for the depth component
            in multi-task scenarios. Defaults to None.
        num_classes_p (int, optional): Number of classes for the 'P' dimension (used in pose_multi
            and multi_tasks). Defaults to 0.
        num_classes_r (int, optional): Number of classes for the 'R' dimension (used in pose_multi
            and multi_tasks). Defaults to 0.
        multi_label (bool, optional): Flag indicating if multi-label classification is used
            (affects output splitting). Defaults to False.
        multi_tasks_regression_loss_weight (float, optional): Weighting factor for the depth
            regression loss in multi-task scenarios. Defaults to 1.0.

    Returns:
        Dict[str, float]: A dictionary containing calculated metrics.
            - Always present:
                - "loss": Average loss over the dataset.
                - "total": Total number of samples processed.
            - If task is 'pose_single', 'pose_multi', or 'multi_tasks':
                - "accuracy": Overall accuracy.
                - "precision": Macro precision.
                - "recall": Macro recall.
                - "f1": Macro F1 score.
            - If task is 'pose_multi' or ('multi_tasks' with multi_label=True):
                - "accuracy_p": Accuracy for the P dimension.
                - "precision_p": Precision for P.
                - "recall_p": Recall for P.
                - "f1_p": F1 for P.
                - "accuracy_r": Accuracy for the R dimension.
                - "precision_r": Precision for R.
                - "recall_r": Recall for R.
                - "f1_r": F1 for R.
            - If task is 'multi_tasks':
                - "rmse": Root Mean Squared Error for the depth component.
            - If task is 'depth':
                - "rmse": Root Mean Squared Error.
    """
    running_loss = 0.0
    running_mse = 0.0
    correct = 0
    total = 0

    correct_p = 0
    correct_r = 0

    all_preds = []
    all_targets = []
    all_preds_p = []
    all_targets_p = []
    all_preds_r = []
    all_targets_r = []

    for outputs, labels in inference_results:
        batch_size = outputs.size(0)
        loss = 0.0

        # --- Loss Calculation ---
        if task == Task.pose_multi:
            label_p = labels[:, 0]
            label_r = labels[:, 1]
            out_p = outputs[:, :num_classes_p]
            out_r = outputs[:, num_classes_p:]

            if criterion:
                loss = criterion(out_p, label_p) + criterion(out_r, label_r)

        elif task == Task.multi_tasks:
            pose_labels, depth_labels = labels
            loss_depth_val = 0.0

            if multi_label:
                out_p = outputs[:, :num_classes_p]
                out_r = outputs[:, num_classes_p : num_classes_p + num_classes_r]
                out_d = outputs[:, -1:]
                label_p = pose_labels[:, 0]
                label_r = pose_labels[:, 1]

                if criterion_depth:
                    loss_depth_val = criterion_depth(out_d, depth_labels)

                if criterion_pose and criterion_depth:
                    loss = (
                        criterion_pose(out_p, label_p)
                        + criterion_pose(out_r, label_r)
                        + loss_depth_val * multi_tasks_regression_loss_weight
                    )
            else:
                out_pose = outputs[:, :-1]
                out_d = outputs[:, -1:]

                if criterion_depth:
                    loss_depth_val = criterion_depth(out_d, depth_labels)

                if criterion_pose and criterion_depth:
                    loss = (
                        criterion_pose(out_pose, pose_labels)
                        + loss_depth_val * multi_tasks_regression_loss_weight
                    )

            if criterion_depth:
                running_mse += loss_depth_val.item() * batch_size

        else:
            # pose_single or depth
            if criterion:
                loss = criterion(outputs, labels)

        if isinstance(loss, torch.Tensor):
            running_loss += loss.item() * batch_size
        elif isinstance(loss, float):
            running_loss += loss * batch_size

        # --- Metrics Calculation ---
        if task == Task.pose_single:
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        elif task == Task.pose_multi:
            out_p = outputs[:, :num_classes_p]
            out_r = outputs[:, num_classes_p:]
            _, pred_p = torch.max(out_p, 1)
            _, pred_r = torch.max(out_r, 1)
            total += batch_size
            correct += (
                ((pred_p == labels[:, 0]) & (pred_r == labels[:, 1])).sum().item()
            )
            correct_p += (pred_p == labels[:, 0]).sum().item()
            correct_r += (pred_r == labels[:, 1]).sum().item()
            all_preds_p.extend(pred_p.cpu().numpy())
            all_targets_p.extend(labels[:, 0].cpu().numpy())
            all_preds_r.extend(pred_r.cpu().numpy())
            all_targets_r.extend(labels[:, 1].cpu().numpy())

            # Combine predictions for overall metrics
            combined_pred = pred_p * num_classes_r + pred_r
            combined_label = labels[:, 0] * num_classes_r + labels[:, 1]
            all_preds.extend(combined_pred.cpu().numpy())
            all_targets.extend(combined_label.cpu().numpy())

        elif task == Task.multi_tasks:
            pose_labels, _ = labels
            if multi_label:
                out_p = outputs[:, :num_classes_p]
                out_r = outputs[:, num_classes_p : num_classes_p + num_classes_r]
                _, pred_p = torch.max(out_p, 1)
                _, pred_r = torch.max(out_r, 1)
                total += pose_labels.size(0)
                correct += (
                    ((pred_p == pose_labels[:, 0]) & (pred_r == pose_labels[:, 1]))
                    .sum()
                    .item()
                )
                correct_p += (pred_p == pose_labels[:, 0]).sum().item()
                correct_r += (pred_r == pose_labels[:, 1]).sum().item()
                all_preds_p.extend(pred_p.cpu().numpy())
                all_targets_p.extend(pose_labels[:, 0].cpu().numpy())
                all_preds_r.extend(pred_r.cpu().numpy())
                all_targets_r.extend(pose_labels[:, 1].cpu().numpy())
            else:
                out_pose = outputs[:, :-1]
                _, pred = torch.max(out_pose, 1)
                total += pose_labels.size(0)
                correct += (pred == pose_labels).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(pose_labels.cpu().numpy())

        elif task == Task.depth:
            total += batch_size

    avg_loss = running_loss / dataset_len if dataset_len > 0 else 0.0

    results = {
        "loss": avg_loss,
        "total": total,
    }

    if task in [Task.pose_single, Task.pose_multi, Task.multi_tasks]:
        results["accuracy"] = correct / total if total > 0 else 0.0
        p, r, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro", zero_division=0
        )
        results["precision"] = float(p)
        results["recall"] = float(r)
        results["f1"] = float(f1)

    if task == Task.pose_multi or (task == Task.multi_tasks and multi_label):
        results["accuracy_p"] = correct_p / total if total > 0 else 0.0
        results["accuracy_r"] = correct_r / total if total > 0 else 0.0

        p_p, r_p, f1_p, _ = precision_recall_fscore_support(
            all_targets_p, all_preds_p, average="macro", zero_division=0
        )
        results["precision_p"] = float(p_p)
        results["recall_p"] = float(r_p)
        results["f1_p"] = float(f1_p)

        p_r, r_r, f1_r, _ = precision_recall_fscore_support(
            all_targets_r, all_preds_r, average="macro", zero_division=0
        )
        results["precision_r"] = float(p_r)
        results["recall_r"] = float(r_r)
        results["f1_r"] = float(f1_r)

    if task == Task.multi_tasks:
        results["rmse"] = (running_mse / total) ** 0.5 if total > 0 else 0.0
    elif task == Task.depth:
        results["rmse"] = avg_loss**0.5

    return results


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: Task,
    criterion: Optional[nn.Module] = None,
    criterion_pose: Optional[nn.Module] = None,
    criterion_depth: Optional[nn.Module] = None,
    num_classes_p: int = 0,
    num_classes_r: int = 0,
    multi_label: bool = False,
    multi_tasks_regression_loss_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluates the model on the given dataloader.
    Wrapper around run_inference and evaluate_metrics.
    """
    inference_results = run_inference(model, dataloader, device, task)
    return evaluate_metrics(
        inference_results,
        task,
        len(dataloader.dataset),
        criterion,
        criterion_pose,
        criterion_depth,
        num_classes_p,
        num_classes_r,
        multi_label,
        multi_tasks_regression_loss_weight,
    )
