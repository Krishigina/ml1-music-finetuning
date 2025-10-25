"""
Конфигурация для обучения моделей классификации музыкальных инструментов
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


@dataclass
class DataConfig:
    """Конфигурация данных"""
    data_path: str = "data/raw"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.2
    classes: List[str] = None
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = ["harp", "piano", "violin"]


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    model_name: str = "efficientnet_b0"  # Модель по умолчанию
    pretrained: bool = True
    num_classes: int = 3
    dropout_rate: float = 0.5
    freeze_backbone: bool = True  # Заморозить backbone на начальных эпохах
    unfreeze_epoch: int = 3  # Эпоха для разморозки


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    # Эпохи обучения
    epochs: int = 10  # Общее количество эпох (для совместимости)
    freeze_epochs: int = 3  # Эпохи с замороженным backbone
    unfreeze_epochs: int = 7  # Эпохи полного дообучения
    learning_rate: float = 1e-3  # Learning rate (для совместимости)
    lr_head: float = 1e-3  # Learning rate для головы классификатора
    lr_full: float = 1e-4  # Learning rate для полного дообучения
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    patience: int = 3 # Для early stopping
    
    # Аугментации
    use_augmentation: bool = True
    rotation_degrees: int = 15
    horizontal_flip_prob: float = 0.5
    color_jitter_strength: float = 0.2
    
    @property
    def total_epochs(self) -> int:
        """Общее количество эпох"""
        return self.freeze_epochs + self.unfreeze_epochs


@dataclass
class ONNXConfig:
    """Конфигурация для экспорта в ONNX"""
    export_onnx: bool = True
    onnx_path: str = "experiments/models/best_model.onnx"
    input_names: List[str] = None
    output_names: List[str] = None
    dynamic_axes: Dict[str, Dict[int, str]] = None
    opset_version: int = 11
    optimize_for_cpu: bool = True  
    
    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["output"]
        if self.dynamic_axes is None:
            self.dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }


@dataclass
class ExperimentConfig:
    """Общая конфигурация эксперимента"""
    # Воспроизводимость
    random_seed: int = 11
    
    # Устройство
    device: str = "auto"  # auto, cpu, cuda
    
    # Логирование
    log_interval: int = 10
    save_model: bool = True
    model_save_path: str = "experiments/models"
    
    # ONNX экспорт
    export_onnx: bool = True
    
    # Лучшие результаты из экспериментов
    best_model_name: str = "efficientnet_b0"

    
    # Конфигурации компонентов
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    onnx: ONNXConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
            # Устанавливаем лучшую модель
            self.model.model_name = self.best_model_name
        if self.training is None:
            self.training = TrainingConfig()
        if self.onnx is None:
            self.onnx = ONNXConfig()
            
        # Автоматическое определение устройства
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# Предустановленные конфигурации для разных моделей
def get_best_config() -> ExperimentConfig:
    """Лучшая конфигурация на основе экспериментов (EfficientNet-B0)"""
    config = ExperimentConfig()
    
    # Лучшие параметры из экспериментов
    config.training.freeze_epochs = 3
    config.training.unfreeze_epochs = 7
    config.training.lr_head = 1e-3
    config.training.lr_full = 1e-4
    config.training.optimizer = "adam"
    config.training.patience = 3
    
    return config


def get_resnet_config() -> ExperimentConfig:
    """Конфигурация для ResNet18"""
    config = ExperimentConfig()
    config.model.model_name = "resnet18"
    
    config.training.freeze_epochs = 5
    config.training.unfreeze_epochs = 10
    config.training.lr_head = 1e-3
    config.training.lr_full = 1e-4
    config.training.optimizer = "adam"
    config.training.patience = 3
    
    return config


def get_efficientnet_config() -> ExperimentConfig:
    """Конфигурация для EfficientNet-B0"""
    config = ExperimentConfig()
    config.model.model_name = "efficientnet_b0"
    
    config.training.freeze_epochs = 5
    config.training.unfreeze_epochs = 10
    config.training.lr_head = 1e-3
    config.training.lr_full = 1e-4
    config.training.optimizer = "adam"
    config.training.patience = 3
    
    return config