from pprint import pprint
from typing import List, Union

import json
import lm_eval.base

from . import CroF
# Clear GPU memory
import torch
torch.cuda.empty_cache()
TASK_REGISTRY = {
    "CroF_fe": CroF.FETask,
    "CroF_fbp": CroF.FpbTask,
    "CroF_FinSent": CroF.FinSentTask,
    "CroF_MayFPB": CroF.MayFPBTask,
    
    "CroF_StockA": CroF.StockATask,
    "CroF_ACL18": CroF.ACL18Task,
    "CroF_IndCIKM18": CroF.StockIndCIKM18,
    "CroF_sm_aclFil": CroF.StockMovementACLFil,
    
    "CroF_NA": CroF.NATask,
    "CroF_EDTSUM": CroF.EDTSUM,
    "CroF_IDFinURLSum": CroF.IDFinURLSumTask,
    "CroF_ThaNA": CroF.ThaNATask,
    
    "CroF_NL": CroF.NLTask,
    "CroF_Headlines": CroF.Headlines,
    "CroF_AppRevs": CroF.AppRevsTask,
    "CroF_VieNL": CroF.VieNLTask,

    **CroF.SM_TASKS,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))

_EXAMPLE_JSON_PATH = "split:key:/absolute/path/to/data.json"


def add_json_task(task_name):
    """Add a JSON perplexity task if the given task name matches the
    JSON task specification.

    See `json.JsonPerplexity`.
    """
    if not task_name.startswith("json"):
        return

    def create_json_task():
        splits = task_name.split("=", 1)
        if len(splits) != 2 or not splits[1]:
            raise ValueError(
                "json tasks need a path argument pointing to the local "
                "dataset, specified like this: json="
                + _EXAMPLE_JSON_PATH
                + ' (if there are no splits, use "train")'
            )

        json_path = splits[1]
        if json_path == _EXAMPLE_JSON_PATH:
            raise ValueError(
                "please do not copy the example path directly, but substitute "
                "it with a path to your local dataset"
            )
        return lambda: json.JsonPerplexity(json_path)

    TASK_REGISTRY[task_name] = create_json_task()


def get_task(task_name):
    try:
        add_json_task(task_name)
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
