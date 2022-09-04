from abc import ABC
from typing import List
import math
import ujson
import os
import random

import torch
from torch.utils.data import DataLoader
from torch.utils import data

from evaluation_network.data.transformer import StateTransformer


def data_loader(folder_path, transformer: StateTransformer, batch_size, shuffle=False, buffer_size=30000, num_workers=0):
    files = sorted(
        [
            os.path.join(folder_path, file_name)
            for file_name in os.listdir(folder_path)
            if file_name.endswith(".jsonl")
        ]
    )[:]

    datasets = [TurnsDataset(file, transformer) for file in files]
    dataset = MultiDataDataset(datasets)

    if shuffle:
        dataset = BufferedShuffleDataset(dataset, buffer_size)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader


class TurnsDataset(data.IterableDataset, ABC):
    def __init__(self, file: str, transform: StateTransformer):
        super().__init__()
        self.file = file
        self.transform = transform

    def __iter__(self):
        with open(self.file, "r") as f:
            for line in f:
                d = ujson.loads(line)
                t = self.transform(d)
                yield t


class MultiDataDataset(data.IterableDataset, ABC):
    def __init__(self, datasets: List[data.IterableDataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self.start = 0
        self.end = len(self.datasets)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for d in self.datasets[iter_start:iter_end]:
            for x in d:
                yield x


class BufferedShuffleDataset(data.IterableDataset):
    """ from Torch 1.8.1

    Dataset shuffled from the original dataset.

        This class is useful to shuffle an existing instance of an IterableDataset.
        The buffer with `buffer_size` is filled with the items from the dataset first. Then,
        each item will be yielded from the buffer by reservoir sampling via iterator.

        `buffer_size` is required to be larger than 0. For `buffer_size == 1`, the
        dataset is not shuffled. In order to fully shuffle the whole dataset, `buffer_size`
        is required to be greater than or equal to the size of dataset.

        When it is used with :class:`~torch.utils.data.DataLoader`, each item in the
        dataset will be yielded from the :class:`~torch.utils.data.DataLoader` iterator.
        And, the method to set up a random seed is different based on :attr:`num_workers`.

        For single-process mode (:attr:`num_workers == 0`), the random seed is required to
        be set before the :class:`~torch.utils.data.DataLoader` in the main process.

            ds = BufferedShuffleDataset(dataset)
            random.seed(...)
            print(list(torch.utils.data.DataLoader(ds, num_workers=0)))

        For multi-process mode (:attr:`num_workers > 0`), the random seed is set by a callable
        function in each worker.

            ds = BufferedShuffleDataset(dataset)
            def init_fn(worker_id):
            ...     random.seed(...)
            print(list(torch.utils.data.DataLoader(ds, ..., num_workers=n, worker_init_fn=init_fn)))

        Args:
            dataset (IterableDataset): The original IterableDataset.
            buffer_size (int): The buffer size for shuffling.
        """
    def __init__(self, dataset: data.IterableDataset, buffer_size: int) -> None:
        super(BufferedShuffleDataset, self).__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


