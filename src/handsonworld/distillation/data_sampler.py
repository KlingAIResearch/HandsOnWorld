"""Resume a distributed epoch without reading and discarding dataset items."""

from itertools import islice

from torch.utils.data import DistributedSampler


class ResumeDistributedSampler(DistributedSampler):
    """Keep DistributedSampler ordering/padding, but start at a local index offset.

    Length reports the remaining samples. Callers tracking training steps must
    retain the full epoch length before setting an offset. The control trainer
    uses batch_size=1, so its batch offset equals this sample offset.
    """

    start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(f"Invalid local sample offset {start_index} for {self.num_samples} samples")
        super().set_epoch(epoch)
        self.start_index = start_index

    def __iter__(self):
        return islice(super().__iter__(), self.start_index, None)

    def __len__(self):
        return self.num_samples - self.start_index
