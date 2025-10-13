from pytorch_lightning.callbacks import LearningRateMonitor
from coda.configs import builds, MainStore


MainStore.store(name="pl", node=builds(LearningRateMonitor), group="callbacks/lr_monitor")
