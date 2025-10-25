"""
Gradio приложение для классификации музыкальных инструментов
Использует ONNX модель для инференса на CPU
"""

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
from pathlib import Path

class MusicInstrumentClassifier:
    def __init__(self, model_path: str):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к ONNX модели
        """
        self.model_path = model_path
        self.session = None
        self.class_names = ['harp', 'piano', 'violin']
        self.class_names_ru = ['Арфа', 'Пианино', 'Скрипка']
        self.input_size = 224
        
        # Параметры нормализации ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка ONNX модели"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            
            # Создаем сессию ONNX Runtime для CPU
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            print(f"Модель загружена: {self.model_path}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.session = None
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Предобработка изображения для модели
        
        Args:
            image: PIL изображение
            
        Returns:
            Предобработанный тензор
        """
        # Конвертируем в RGB если необходимо
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Изменяем размер
        image = image.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
        
        # Конвертируем в numpy array и нормализуем
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Применяем нормализацию ImageNet
        mean = self.mean.astype(np.float32)
        std = self.std.astype(np.float32)
        img_array = (img_array - mean) / std
        
        # Изменяем размерность: (H, W, C) -> (1, C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Убеждаемся что результат float32
        return img_array.astype(np.float32)
    
    def predict(self, image: Image.Image) -> dict:
        """
        Предсказание класса изображения
        
        Args:
            image: PIL изображение
            
        Returns:
            Словарь с вероятностями для каждого класса
        """
        if self.session is None:
            return {"Модель не загружена": 1.0}
        
        try:
            input_tensor = self._preprocess_image(image)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            predictions = outputs[0][0]
            
            # Softmax
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / np.sum(exp_preds)
            
            results = {}
            for class_name, class_name_ru, prob in zip(self.class_names, self.class_names_ru, probabilities):
                results[f"{class_name_ru} ({class_name})"] = float(prob)
            
            return results
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return {"Ошибка": 1.0}

def find_best_model() -> str:
    """Поиск лучшей ONNX модели в папке experiments/models"""
    models_dir = Path("../experiments/models")
    
    if not models_dir.exists():
        return None
    
    # Ищем модели в порядке приоритета
    priority_models = ["best_model.onnx", "model.onnx", "resnet18.onnx", "efficientnet_b0.onnx"]
    
    for model_name in priority_models:
        model_path = models_dir / model_name
        if model_path.exists():
            return str(model_path)
    
    # Если приоритетные модели не найдены, берем любую ONNX
    onnx_files = list(models_dir.glob("*.onnx"))
    
    if not onnx_files:
        return None
    
    return str(onnx_files[0])

def create_interface():
    """Создание интерфейса Gradio"""
    model_path = find_best_model()
    
    if model_path is None:
        def no_model_predict(image):
            return {"Модель не найдена": 1.0}
        
        interface = gr.Interface(
            fn=no_model_predict,
            inputs=gr.Image(type="pil", label="Изображение"),
            outputs=gr.Label(num_top_classes=3, label="Результат"),
            title="Классификация инструментов",
            description="Загрузите изображение (арфа, пианино, скрипка).",
            allow_flagging="never"
        )
        return interface
    
    classifier = MusicInstrumentClassifier(model_path)
    
    interface = gr.Interface(
        fn=classifier.predict,
        inputs=gr.Image(type="pil", label="Изображение"),
        outputs=gr.Label(num_top_classes=3, label="Результат"),
        title="Классификация инструментов",
        description="Загрузите изображение (арфа, пианино, скрипка).",
        allow_flagging="never"
    )
    
    return interface

def main():
    """Главная функция запуска приложения"""
    interface = create_interface()
    port = int(os.environ.get('GRADIO_SERVER_PORT', 7863))
    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()