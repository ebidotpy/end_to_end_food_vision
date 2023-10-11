from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataLoaderConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    image_shape: tuple
    batch_size: int
    class_mode: str

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    include_top: bool
    trainable: bool
    input_shape: list
    freeze_all: bool
    freeze_till: int
    learning_rate: float
    classes: int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    train_dir: Path
    test_dir: Path
    epochs: int

@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path
    score_path: Path