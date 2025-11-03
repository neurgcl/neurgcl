import math
import warnings
from typing import Iterator, List, Optional, Sequence, Sized, TypeVar, Union

import torch

# No 'default_generator' in torch/__init__.pyi
from torch import Generator, default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import BatchSampler, Dataset, RandomSampler, Sampler

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[torch.tensor(self.indices)[idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split_slice(
    dataset: Dataset[T], lengths: Sequence[Union[int, float]], generator: Optional[Generator] = default_generator
) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(math.floor(len(dataset) * frac))  # type: ignore[arg-type]
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. " f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class SliceBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, n, batch_size: int, shuffle: bool = False, drop_last: bool = False, generator=None) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.generator = generator
        self.shuffle = shuffle
        self.n = n

        if not self.shuffle:
            self.idx_list = list(range(self.n))

    def __iter__(self) -> Iterator[List[int]]:
        n = self.n

        if self.shuffle:
            if self.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
            else:
                generator = self.generator
            idx_list = torch.randperm(n, generator=generator).tolist()
        else:
            idx_list = self.idx_list

        if self.drop_last:
            for i in range(0, n, self.batch_size):
                if i != 0 and (i + self.batch_size) > n:
                    break
                idx_batch = idx_list[i : i + self.batch_size]
                yield idx_batch
        else:
            for i in range(0, n, self.batch_size):
                idx_batch = idx_list[i : i + self.batch_size]
                yield idx_batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return self.n // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.n + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class SliceBatchLoader(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self, dataset: Sized, batch_size: int, shuffle: bool = False, drop_last: bool = False, generator=None
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))

        self.batch_size = batch_size
        self.drop_last = drop_last

        self.generator = generator
        self.shuffle = shuffle
        self.dataset = dataset
        self.whole_ds = self.batch_size >= len(self.dataset)
        self.n = len(self.dataset)

        if self.whole_ds:
            self.__iter__ = self.yield_whole_ds

        if not self.shuffle:
            self.idx_list = list(range(self.n))

    def yield_whole_ds(self):
        yield (self.dataset)

    def __iter__(self) -> Iterator[List[int]]:
        n = self.n

        if self.shuffle:
            if self.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
            else:
                generator = self.generator
            idx_list = torch.randperm(n, generator=generator)

            if self.drop_last:
                for i in range(0, n, self.batch_size):
                    if i != 0 and (i + self.batch_size) > n:
                        break
                    idx_batch = idx_list[i : i + self.batch_size]
                    yield (self.dataset[idx_batch])
            else:
                for i in range(0, n, self.batch_size):
                    idx_batch = idx_list[i : i + self.batch_size]
                    yield (self.dataset[idx_batch])
        else:
            for i in range(0, n, self.batch_size):
                if self.drop_last and i != 0 and (i + self.batch_size) > n:
                    break
                yield (self.dataset[i : i + self.batch_size])

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return self.n // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.n + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
