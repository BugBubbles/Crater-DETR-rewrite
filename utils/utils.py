from functools import partial
from mmdet.models.utils import multi_apply
import torch


def multi_apply_dict(func, *args, **kwargs):
    """Apply function to a dict of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = list(map(pfunc, *args))
    if len(map_results) == 0:
        return dict()
    keys = map_results[0].keys()
    values = map(torch.stack, multi_apply(lambda x: x.values(), map_results))
    return dict(zip(keys, values))
