from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ReduceLROnPlateau,
    StepLR,
)

from src.utils.mapper import ConfigMapper

ConfigMapper.map("schedulers", "step")(StepLR)
ConfigMapper.map("schedulers", "cosineanneal")(CosineAnnealingLR)
ConfigMapper.map("schedulers", "reduceplateau")(ReduceLROnPlateau)
ConfigMapper.map("schedulers", "cyclic")(CyclicLR)
ConfigMapper.map("schedulers", "cosineannealrestart")(
    CosineAnnealingWarmRestarts
)
