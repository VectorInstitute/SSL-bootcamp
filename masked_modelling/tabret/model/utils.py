"""Tabret model utils."""
from typing import Dict, List, Tuple

from torch import nn


def get_parameter_names(
    model: nn.Module,
    forbidden_layer_types: List[nn.Module],
) -> List[str]:
    """Return the names of the parameters not inside a forbidden layer."""
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter)
    # since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_diff_columns(
    continuous_columns: List[str],
    cat_cardinality_dict: Dict[str, int],
    pre_continuous_columns: List[str],
    pre_cat_cardinality_dict: Dict[str, int],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Find the difference between the pretraining and finetuning columns."""
    continuous_columns = [
        col for col in continuous_columns if col not in pre_continuous_columns
    ]
    cat_cardinality_dict = {
        col: item
        for col, item in cat_cardinality_dict.items()
        if col not in pre_cat_cardinality_dict
    }

    diff_columns = continuous_columns + list(cat_cardinality_dict.keys())

    return diff_columns, continuous_columns, cat_cardinality_dict
