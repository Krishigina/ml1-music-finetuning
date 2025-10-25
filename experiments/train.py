"""
Обучение модели классификации музыкальных инструментов
с использованием лучших параметров из экспериментов и экспортом в ONNX
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse
from typing import Tuple, Dict, Any
import logging

# Добавляем текущую папку для импорта конфигураций
sys.path.append(str(Path(__file__).parent))

from config import ExperimentConfig, get_best_config, get_resnet_config, get_efficientnet_config
from config import TransformedDataset
from torchvision import datasets, transforms


def setup_logging(log_level: str = "INFO") -> None:
    """Настройка логирования"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def set_seed(seed: int) -> None:
    """Установка seed для воспроизводимости"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(config: ExperimentConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Создание трансформаций для обучения и валидации"""
    # Нормализация ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Трансформации для обучения
    train_transforms = [
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        normalize
    ]
    
    if config.training.use_augmentation:
        train_transforms.insert(-2, transforms.RandomRotation(config.training.rotation_degrees))
        train_transforms.insert(-2, transforms.RandomHorizontalFlip(config.training.horizontal_flip_prob))
        train_transforms.insert(-2, transforms.ColorJitter(
            brightness=config.training.color_jitter_strength,
            contrast=config.training.color_jitter_strength,
            saturation=config.training.color_jitter_strength,
            hue=config.training.color_jitter_strength/2
        ))
    
    # Трансформации для валидации
    val_transforms = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return transforms.Compose(train_transforms), val_transforms


def create_data_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """Создание загрузчиков данных"""
    train_transform, val_transform = get_transforms(config)
    
    # Загрузка полного датасета
    full_dataset = datasets.ImageFolder(root=config.data.data_path)
    
    # Разделение на train/val
    train_size = int(config.data.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.random_seed)
    )
    
    # Применение трансформаций
    train_dataset = TransformedDataset(train_dataset, train_transform)
    val_dataset = TransformedDataset(val_dataset, val_transform)
    
    # Создание загрузчиков
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return train_loader, val_loader


def create_model(config: ExperimentConfig) -> nn.Module:
    """Создание модели"""
    model = timm.create_model(
        config.model.model_name,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes
    )
    
    # Добавление dropout если указан
    if hasattr(model, 'classifier') and config.model.dropout_rate > 0:
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(in_features, config.model.num_classes)
        )
    
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Заморозка backbone модели"""
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'head' not in name and 'fc' not in name:
            param.requires_grad = False
    logging.info("Backbone заморожен")


def unfreeze_backbone(model: nn.Module) -> None:
    """Разморозка backbone модели"""
    for param in model.parameters():
        param.requires_grad = True
    logging.info("Backbone разморожен")


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: str, epoch: int, log_interval: int) -> float:
    """Обучение одной эпохи"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: str) -> Tuple[float, float]:
    """Валидация модели"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = correct / total
    
    return val_loss, val_acc


def export_to_onnx(model: nn.Module, config: ExperimentConfig, device: str) -> None:
    """Экспорт модели в ONNX формат для CPU инференса"""
    if not config.onnx.export_onnx:
        return
    
    logging.info("Начинаем экспорт модели в ONNX...")
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(config.onnx.onnx_path), exist_ok=True)
    
    # Переводим модель в режим eval и на CPU для экспорта
    model.eval()
    model_cpu = model.cpu()
    
    # Создаем dummy input
    dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    
    try:
        torch.onnx.export(
            model_cpu,
            dummy_input,
            config.onnx.onnx_path,
            export_params=True,
            opset_version=config.onnx.opset_version,
            do_constant_folding=True,
            input_names=config.onnx.input_names,
            output_names=config.onnx.output_names,
            dynamic_axes=config.onnx.dynamic_axes
        )
        
        logging.info(f"Модель успешно экспортирована в ONNX: {config.onnx.onnx_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(config.onnx.onnx_path) / (1024 * 1024)  # MB
        logging.info(f"Размер ONNX файла: {file_size:.2f} MB")
        
    except Exception as e:
        logging.error(f"Ошибка при экспорте в ONNX: {e}")
        raise


def plot_training_curves(train_losses: list, val_losses: list, train_accs: list, 
                        val_accs: list, save_path: str) -> None:
    """Построение графиков обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_model_with_fine_tuning(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Основная функция обучения с двухэтапным дообучением:
    1. Этап 1: Обучение только головы классификатора (backbone заморожен)
    2. Этап 2: Полное дообучение всей модели
    """
    logging.info("=== НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ ===")
    logging.info(f"Модель: {config.model.model_name}")
    logging.info(f"Устройство: {config.device}")
    logging.info(f"Этап 1 (заморозка): {config.training.freeze_epochs} эпох")
    logging.info(f"Этап 2 (полное дообучение): {config.training.unfreeze_epochs} эпох")
    
    # Установка seed
    set_seed(config.random_seed)
    
    # Создание загрузчиков данных
    train_loader, val_loader = create_data_loaders(config)
    logging.info(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    logging.info(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    
    # Создание модели
    model = create_model(config)
    model = model.to(config.device)
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Общее количество параметров: {total_params:,}")
    logging.info(f"Обучаемых параметров: {trainable_params:,}")
    
    # Критерий потерь
    criterion = nn.CrossEntropyLoss()
    
    # История обучения
    history = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': [],
        'best_val_acc': 0.0, 'best_epoch': 0
    }
    
    # === ЭТАП 1: ОБУЧЕНИЕ ГОЛОВЫ КЛАССИФИКАТОРА ===
    logging.info("\n=== ЭТАП 1: ОБУЧЕНИЕ ГОЛОВЫ КЛАССИФИКАТОРА ===")
    freeze_backbone(model)
    
    # Оптимизатор для головы
    optimizer_head = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.lr_head,
        weight_decay=config.training.weight_decay
    )
    
    for epoch in range(1, config.training.freeze_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer_head, criterion, 
            config.device, epoch, config.log_interval
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        logging.info(f"Epoch {epoch}/{config.training.freeze_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
    
    # === ЭТАП 2: ПОЛНОЕ ДООБУЧЕНИЕ ===
    logging.info("\n=== ЭТАП 2: ПОЛНОЕ ДООБУЧЕНИЕ ===")
    unfreeze_backbone(model)
    
    # Новый оптимизатор для всей модели
    optimizer_full = optim.Adam(
        model.parameters(),
        lr=config.training.lr_full,
        weight_decay=config.training.weight_decay
    )
    
    # Early stopping
    patience_counter = 0
    
    for epoch in range(config.training.freeze_epochs + 1, 
                      config.training.freeze_epochs + config.training.unfreeze_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer_full, criterion, 
            config.device, epoch, config.log_interval
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        logging.info(f"Epoch {epoch}/{config.training.total_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Сохранение лучшей модели
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch
            patience_counter = 0
            
            # Сохранение лучшей модели
            if config.save_model:
                os.makedirs(config.model_save_path, exist_ok=True)
                best_model_path = os.path.join(config.model_save_path, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_full.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, best_model_path)
                logging.info(f"Лучшая модель сохранена: {best_model_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config.training.patience:
            logging.info(f"Early stopping на эпохе {epoch}")
            break
    
    # === ФИНАЛИЗАЦИЯ ===
    logging.info(f"\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    logging.info(f"Лучшая точность на валидации: {history['best_val_acc']:.4f} на эпохе {history['best_epoch']}")
    
    # Загрузка лучшей модели для экспорта
    if config.save_model:
        best_model_path = os.path.join(config.model_save_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Загружена лучшая модель для экспорта")
    
    # Экспорт в ONNX
    export_to_onnx(model, config, config.device)
    
    # Построение графиков
    if config.save_model:
        plot_path = os.path.join(config.model_save_path, 'training_curves.png')
        plot_training_curves(
            history['train_losses'], history['val_losses'],
            history['train_accs'], history['val_accs'], plot_path
        )
        logging.info(f"Графики обучения сохранены: {plot_path}")
    
    return history


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Обучение модели классификации музыкальных инструментов')
    parser.add_argument('--config', type=str, default='best', 
                       choices=['best', 'resnet', 'efficientnet'],
                       help='Конфигурация для использования')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Путь к данным (переопределяет конфигурацию)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Устройство для обучения')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Уровень логирования')
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging(args.log_level)
    
    # Выбор конфигурации
    if args.config == 'best':
        config = get_best_config()
    elif args.config == 'resnet':
        config = get_resnet_config()
    elif args.config == 'efficientnet':
        config = get_efficientnet_config()
    else:
        raise ValueError(f"Неизвестная конфигурация: {args.config}")
    
    # Переопределение параметров из командной строки
    if args.data_path:
        config.data.data_path = args.data_path
    if args.device != 'auto':
        config.device = args.device
    
    logging.info(f"Используется конфигурация: {args.config}")
    logging.info(f"Путь к данным: {config.data.data_path}")
    logging.info(f"Устройство: {config.device}")
    
    # Проверка существования данных
    if not os.path.exists(config.data.data_path):
        logging.error(f"Путь к данным не существует: {config.data.data_path}")
        return
    
    # Запуск обучения
    try:
        history = train_model_with_fine_tuning(config)
        logging.info("Обучение успешно завершено!")
        
        # Вывод итоговых результатов
        print(f"\n{'='*50}")
        print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print(f"{'='*50}")
        print(f"Модель: {config.model.model_name}")
        print(f"Лучшая точность на валидации: {history['best_val_acc']:.4f}")
        print(f"Достигнута на эпохе: {history['best_epoch']}")
        print(f"Общее количество эпох: {len(history['train_losses'])}")
        if config.onnx.export_onnx:
            print(f"ONNX модель сохранена: {config.onnx.onnx_path}")
        print(f"{'='*50}")
        
    except Exception as e:
        logging.error(f"Ошибка при обучении: {e}")
        raise


if __name__ == "__main__":
    main()