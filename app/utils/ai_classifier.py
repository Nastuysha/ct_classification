import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import logging
from pathlib import Path

class CTClassificationModel(nn.Module):
    """CNN модель для классификации КТ исследований ОГК"""
    
    def __init__(self, num_classes=2):
        super(CTClassificationModel, self).__init__()
        
        # Основная CNN архитектура
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling слои
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Первый блок
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Второй блок
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Третий блок
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Четвертый блок
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class CTClassifier:
    """Классификатор КТ исследований ОГК"""
    
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_version = "v1.0.0"
        
        # Загрузка модели
        self._load_model(model_path)
        
    def _load_model(self, model_path: str = None):
        """Загрузка предобученной модели"""
        try:
            self.model = CTClassificationModel()
            
            if model_path and Path(model_path).exists():
                # Загрузка весов модели
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_version = checkpoint.get('version', self.model_version)
                self.logger.info(f"Модель загружена из {model_path}")
            else:
                # Инициализация случайными весами для MVP
                self._initialize_weights()
                self.logger.info("Модель инициализирована случайными весами")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {e}")
            # Fallback к случайным весам
            self.model = CTClassificationModel()
            self._initialize_weights()
            self.model.to(self.device)
            self.model.eval()
    
    def _initialize_weights(self):
        """Инициализация весов модели"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def classify_images(self, images: List[np.ndarray]) -> Tuple[str, float]:
        """
        Классификация серии изображений КТ
        
        Args:
            images: Список предобработанных изображений
            
        Returns:
            Tuple[str, float]: (результат классификации, уверенность)
        """
        if not images:
            return "error", 0.0
        
        try:
            # Подготовка данных
            batch_tensor = self._prepare_batch(images)
            
            # Инференс
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Агрегация результатов по всем изображениям
                avg_probabilities = torch.mean(probabilities, dim=0)
                
                # Получение результата
                predicted_class = torch.argmax(avg_probabilities).item()
                confidence = avg_probabilities[predicted_class].item()
                
                # Преобразование в читаемый результат
                result = "normal" if predicted_class == 0 else "pathology"
                
                self.logger.info(f"Классификация завершена: {result}, уверенность: {confidence:.3f}")
                
                return result, confidence
                
        except Exception as e:
            self.logger.error(f"Ошибка при классификации: {e}")
            return "error", 0.0
    
    def _prepare_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Подготовка батча изображений для модели"""
        # Конвертация в тензоры
        tensors = []
        for img in images:
            # Изменение порядка осей (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            tensors.append(img_tensor)
        
        # Создание батча
        batch = torch.stack(tensors)
        return batch.to(self.device)
    
    def get_model_version(self) -> str:
        """Получение версии модели"""
        return self.model_version
    
    def save_model(self, save_path: str):
        """Сохранение модели"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'version': self.model_version
            }
            torch.save(checkpoint, save_path)
            self.logger.info(f"Модель сохранена в {save_path}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении модели: {e}")
