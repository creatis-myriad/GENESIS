import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_epochs: int,
    num_training_epochs: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Generate a learning rate scheduler that implements cosine annealing after an initial linear warmup.

    References:
         - Original implementation by Huggingface:
           https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/optimization.py#L104-L135

    Args:
        optimizer: Wrapper optimizer.
        num_warmup_epochs: Number of epochs for the warmup phase, during which the learning rate increases linearly from
            0 to the initial learning rate.
        num_training_epochs: Total number of training epochs.
        num_cycles: Number of cycles (i.e. waves) in the cosine annealing schedule.
            Defaults is to just decrease from the max value to 0 following a half-cosine.
        last_epoch: Index of the last epoch (when resuming training).

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_epochs:
            return float(current_step) / float(max(1, num_warmup_epochs))
        progress = float(current_step - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
