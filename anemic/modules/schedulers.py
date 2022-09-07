from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
)
from transformers import get_linear_schedule_with_warmup

from anemic.utils.mapper import ConfigMapper

ConfigMapper.map("schedulers", "step")(StepLR)
ConfigMapper.map("schedulers", "cosineanneal")(CosineAnnealingLR)
ConfigMapper.map("schedulers", "reduceplateau")(ReduceLROnPlateau)
ConfigMapper.map("schedulers", "cyclic")(CyclicLR)
ConfigMapper.map("schedulers", "cosineannealrestart")(
    CosineAnnealingWarmRestarts
)
ConfigMapper.map("schedulers", "linearwithwarmup")(
    get_linear_schedule_with_warmup
)
