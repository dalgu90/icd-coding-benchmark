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


ConfigMapper.map("schedulers", "step")(StepLR)
ConfigMapper.map("schedulers", "cosineanneal")(CosineAnnealingLR)
ConfigMapper.map("schedulers", "reduceplateau")(ReduceLROnPlateau)
ConfigMapper.map("schedulers", "cyclic")(CyclicLR)
ConfigMapper.map("schedulers", "cosineannealrestart")(
    CosineAnnealingWarmRestarts
)
ConfigMapper.map("schedulers", "linearwithwarmup")(get_linear_schedule_with_warmup)
