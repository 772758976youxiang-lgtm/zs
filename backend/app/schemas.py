from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DatasetCreate(Strict):
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["pretrain", "finetune"] = "pretrain"
    rule_version: str = Field(default="unit-v1", min_length=1, max_length=64)
    records: list[dict] = Field(min_length=1, max_length=100000)
    base_dataset_id: str | None = None


class ModelConfig(Strict):
    hidden_size: int = Field(default=768, ge=16, le=1024)
    num_layers: int = Field(default=14, ge=1, le=24)
    num_heads: int = Field(default=12, ge=1, le=32)
    intermediate_size: int = Field(default=3072, ge=32, le=4096)
    max_length: int = Field(default=512, ge=8, le=2048)
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def divisible(self):
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        return self


class ModelCreate(Strict):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    dataset_id: str
    preset: Literal["100m", "debug"] = "100m"
    seed: int = Field(default=42, ge=0, le=2147483647)
    config: ModelConfig | None = None


class TrainConfig(Strict):
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    micro_batch: int = Field(default=2, ge=1, le=64)
    gradient_accumulation: int = Field(default=1, ge=1, le=128)
    max_steps: int = Field(default=100, ge=1, le=10000000)
    save_every: int = Field(default=25, ge=1, le=100000)
    eval_every: int = Field(default=10, ge=1, le=100000)
    mask_ratio: float = Field(default=0.15, gt=0, le=1)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    gradient_clip: float = Field(default=1, gt=0, le=100)
    activation_checkpointing: bool = False
    device: Literal["cuda", "cpu"] = "cuda"
    seed: int = Field(default=42, ge=0, le=2147483647)
    sequence_length: int = Field(default=128, ge=8, le=2048)


class RunCreate(Strict):
    model_id: str
    dataset_id: str
    task: Literal["pretrain", "finetune"]
    parent_checkpoint_id: str | None = None
    demo_overfit: bool = False
    config: TrainConfig = Field(default_factory=TrainConfig)


class MLMRequest(Strict):
    checkpoint_id: str
    text: str = Field(max_length=10000)
    mask_positions: list[int]
    top_k: int = Field(default=5, ge=1, le=20)


class UnitsRequest(Strict):
    checkpoint_id: str
    text: str = Field(max_length=10000)
    threshold: float = Field(default=0.5, ge=0, le=1)


class Output(BaseModel):
    model_config = ConfigDict(extra="allow")


class DatasetInfo(Output):
    id: str
    name: str
    kind: Literal["pretrain", "finetune"]
    rule_version: str
    hash: str
    count: int
    counts: dict[str, int]
    confirmed_count: int
    characters: int
    vocab_size: int
    warning: str | None = None


class LayerInfo(Output):
    name: str
    parameters: int


class ModelInfo(Output):
    id: str
    name: str
    preset: Literal["100m", "debug"]
    parameter_count: int
    config: ModelConfig
    layers: list[LayerInfo]
    vocab: list[str]
    validation_coverage: float | None = None


class RunInfo(Output):
    id: str
    task: Literal["pretrain", "finetune"]
    status: str
    model_id: str
    dataset_id: str
    global_step: int
    metrics: dict
    config: TrainConfig
    error: str | None = None
    last_checkpoint_id: str | None = None
    best_checkpoint_id: str | None = None
    rule_version: str
    dataset_hash: str


class CheckpointInfo(Output):
    id: str
    run_id: str
    task: Literal["pretrain", "finetune"]
    model_id: str
    global_step: int
    is_best: bool = False
