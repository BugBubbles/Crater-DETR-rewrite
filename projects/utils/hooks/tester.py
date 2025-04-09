"""test after each validate epoch"""

from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import DataLoader
from mmengine.hooks import Hook
from mmengine.runner import BaseLoop, TestLoop
from mmdet.registry import HOOKS, LOOPS, EVALUATOR, DATASETS, DATA_SAMPLERS
from mmengine.registry import FUNCTIONS
import copy
from mmengine.utils import digit_version
from mmengine.logging import print_log
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.runner.utils import _get_batch_size
from mmengine.runner.runner import _SlicedDataset
from mmengine.dist import get_world_size, get_rank
import logging
from mmengine.evaluator import Evaluator


@HOOKS.register_module()
class ValTestHook(Hook):
    """Test models after each ValLoop"""

    def __init__(
        self,
        loop: Union[BaseLoop, Dict],
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
    ) -> None:
        self._loop = loop
        self._test_dataloader = self.build_dataloader(dataloader)
        self._test_evaluator = self.build_evaluator(evaluator)

    def before_run(self, runner):
        # the ruuner was referenced in the loop, so its model weights
        # will be updated after each val epoch
        self._loop = self.build_test_loop(runner, self._loop)

    def after_val_epoch(self, runner, metrics=None):
        self._loop.run()

    def build_test_loop(self, runner, loop: Union[BaseLoop, Dict]) -> TestLoop:
        """Build test loop.

        Examples of ``loop``::

            # `TestLoop` will be used
            loop = dict()

            # custom test loop
            loop = dict(type='CustomTestLoop')

        Args:
            loop (BaseLoop or dict): A test loop or a dict to build test loop.
                If ``loop`` is a test loop object, just returns itself.

        Returns:
            :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f"test_loop should be a Loop object or dict, but got {type(loop)}"
            )

        loop_cfg = copy.deepcopy(loop)  # type: ignore

        if "type" in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=runner,
                    dataloader=self._test_dataloader,
                    evaluator=self._test_evaluator,
                ),
            )
        else:
            loop = TestLoop(
                **loop_cfg,
                runner=runner,
                dataloader=self._test_dataloader,
                evaluator=self._test_evaluator,
            )  # type: ignore

        return loop  # type: ignore

    def build_evaluator(self, evaluator: Union[Dict, List, Evaluator]) -> Evaluator:
        """Build evaluator.

        Examples of ``evaluator``::

            # evaluator could be a built Evaluator instance
            evaluator = Evaluator(metrics=[ToyMetric()])

            # evaluator can also be a list of dict
            evaluator = [
                dict(type='ToyMetric1'),
                dict(type='ToyEvaluator2')
            ]

            # evaluator can also be a list of built metric
            evaluator = [ToyMetric1(), ToyMetric2()]

            # evaluator can also be a dict with key metrics
            evaluator = dict(metrics=ToyMetric())
            # metric is a list
            evaluator = dict(metrics=[ToyMetric()])

        Args:
            evaluator (Evaluator or dict or list): An Evaluator object or a
                config dict or list of config dict used to build an Evaluator.

        Returns:
            Evaluator: Evaluator build from ``evaluator``.
        """
        if isinstance(evaluator, Evaluator):
            return evaluator
        elif isinstance(evaluator, dict):
            # if `metrics` in dict keys, it means to build customized evalutor
            if "metrics" in evaluator:
                evaluator.setdefault("type", "Evaluator")
                return EVALUATOR.build(evaluator)
            # otherwise, default evalutor will be built
            else:
                return Evaluator(evaluator)  # type: ignore
        elif isinstance(evaluator, list):
            # use the default `Evaluator`
            return Evaluator(evaluator)  # type: ignore
        else:
            raise TypeError(
                "evaluator should be one of dict, list of dict, and Evaluator"
                f", but got {evaluator}"
            )

    @staticmethod
    def build_dataloader(
        dataloader: Union[DataLoader, Dict],
        seed: Optional[int] = None,
        diff_rank_seed: bool = False,
    ) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks.


        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop("dataset")
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, "full_init"):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        num_batch_per_epoch = dataloader_cfg.pop("num_batch_per_epoch", None)
        if num_batch_per_epoch is not None:
            world_size = get_world_size()
            num_samples = (
                num_batch_per_epoch * _get_batch_size(dataloader_cfg) * world_size
            )
            dataset = _SlicedDataset(dataset, num_samples)

        # build sampler
        sampler_cfg = dataloader_cfg.pop("sampler")
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg, default_args=dict(dataset=dataset, seed=sampler_seed)
            )
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop("batch_sampler", None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler, batch_size=dataloader_cfg.pop("batch_size")
                ),
            )
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if "worker_init_fn" in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop("worker_init_fn")
            worker_init_fn_type = worker_init_fn_cfg.pop("type")
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    "type of worker_init_fn should be string or callable "
                    f"object, but got {type(worker_init_fn_type)}"
                )
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn, **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    "disable_subprocess_warning", False
                )
                assert isinstance(disable_subprocess_warning, bool), (
                    "disable_subprocess_warning should be a bool, but got "
                    f"{type(disable_subprocess_warning)}"
                )
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get("num_workers"),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning,
                )
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if "persistent_workers" in dataloader_cfg and digit_version(
            TORCH_VERSION
        ) < digit_version("1.7.0"):
            print_log(
                "`persistent_workers` is only available when " "pytorch version >= 1.7",
                logger="current",
                level=logging.WARNING,
            )
            dataloader_cfg.pop("persistent_workers")

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop("collate_fn", dict(type="pseudo_collate"))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop("type")
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                "collate_fn should be a dict or callable object, but got "
                f"{collate_fn_cfg}"
            )
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg,
        )
        return data_loader
