from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class DatasetConfig(BaseModel):
    name: str
    prompt_name: str
    num_classes: int
    params: Dict[str, Any]

class MetricsLearningModule(BaseModel):
    module_name: str
    params: Dict[str, Any]

class LLMConfig(BaseModel):
    name: str
    backbone: str
    num_classes: int
    use_metrics_learning: bool
    metrics_learning_module: MetricsLearningModule
    cls_head: bool
    seq_head: bool
    params: Optional[Dict[str, Any]] = {}

class LossConfig(BaseModel):
    name: str
    params: Dict[str, float]

class LRSchedulerConfig(BaseModel):
    name: str
    interval: str
    params: Dict[str, float]

class OptimizerConfig(BaseModel):
    name: str
    params: Dict[str, float]

class ExperimentConfig(BaseModel):
    exp_type: str
    exp_name: str
    seed: int
    n_folds: int
    num_classes: int
    max_epochs: int
    check_val_every_n_epoch: int
    sync_batchnorm: bool
    use_lora: bool=False
    precision: str
    num_workers: int
    train_batch_size: int
    accumulate_grad_batchs: int
    valid_batch_size: int
    test_batch_size: int
    dataset_config: DatasetConfig
    llm_config: LLMConfig
    loss_config: LossConfig
    lr_scheduler_config: LRSchedulerConfig
    optimizer_config: OptimizerConfig
    logger: str