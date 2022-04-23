from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR
)

from src.utils.mapper import ConfigMapper
from transformers import get_linear_schedule_with_warmup


def get_linear_schedule_with_warmup(optimizer, num_training_steps, warm_up_proportion = 0.1, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    num_warmup_steps = warm_up_proportion * num_training_steps
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

ConfigMapper.map("schedulers", "step")(StepLR)
ConfigMapper.map("schedulers", "cosineanneal")(CosineAnnealingLR)
ConfigMapper.map("schedulers", "reduceplateau")(ReduceLROnPlateau)
ConfigMapper.map("schedulers", "cyclic")(CyclicLR)
ConfigMapper.map("schedulers", "cosineannealrestart")(
    CosineAnnealingWarmRestarts
)
ConfigMapper.map("schedulers", "linearwithwarmup")(get_linear_schedule_with_warmup)
