from enum import StrEnum


class Task(StrEnum):
    pose_single = "pose_single"
    pose_multi = "pose_multi"
    depth = "depth"
    multi_tasks = "multi_tasks"
