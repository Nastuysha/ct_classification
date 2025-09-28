import pydicom
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import logging

class DicomProcessor:
    """Класс для обработки DICOM файлов КТ исследований ОГК"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_study(self, dicom_directory: Path) -> List[np.ndarray]:
        """
        Обработка всех DICOM файлов в директории исследования
        
        Args:
            dicom_directory: Путь к папке с DICOM файлами
            
        Returns:
            List[np.ndarray]: Список обработанных изображений
        """
        processed_images = []
        
        if not dicom_directory.exists():
            self.logger.error(f"Директория {dicom_directory} не существует")
            return processed_images
        
        # Получение списка DICOM файлов
        dicom_files = list(dicom_directory.glob("*.dcm")) + list(dicom_directory.glob("*.dicom"))
        
        if not dicom_files:
            self.logger.error("DICOM файлы не найдены")
            return processed_images
        
        # Сортировка файлов по номеру среза
        dicom_files = self._sort_dicom_files(dicom_files)
        
        for dicom_file in dicom_files:
            try:
                # Загрузка DICOM файла
                ds = pydicom.dcmread(str(dicom_file))
                
                # Извлечение изображения
                image = self._extract_image(ds)
                
                if image is not None:
                    # Предобработка изображения
                    processed_image = self._preprocess_image(image, ds)
                    processed_images.append(processed_image)
                    
            except Exception as e:
                self.logger.error(f"Ошибка при обработке файла {dicom_file}: {e}")
                continue
        
        self.logger.info(f"Обработано {len(processed_images)} изображений из {len(dicom_files)} файлов")
        return processed_images
    
    def _sort_dicom_files(self, dicom_files: List[Path]) -> List[Path]:
        """Сортировка DICOM файлов по номеру среза"""
        def get_slice_number(file_path):
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                return getattr(ds, 'InstanceNumber', 0)
            except:
                return 0
        
        return sorted(dicom_files, key=get_slice_number)
    
    def _extract_image(self, ds: pydicom.Dataset) -> np.ndarray:
        """Извлечение изображения из DICOM файла"""
        try:
            # Получение пиксельных данных
            pixel_array = ds.pixel_array
            
            # Нормализация HU значений
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            return pixel_array.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении изображения: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
        """
        Предобработка изображения для классификации
        
        Args:
            image: Исходное изображение
            ds: DICOM dataset
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        # Ограничение HU значений для ОГК (от -1000 до 400)
        image = np.clip(image, -1000, 400)
        
        # Нормализация к диапазону [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Изменение размера до стандартного (512x512)
        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Конвертация в 3-канальное изображение для CNN
        image = np.stack([image] * 3, axis=-1)
        
        return image
    
    def create_3d_volume(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Создание 3D объема из серии 2D изображений
        
        Args:
            images: Список 2D изображений
            
        Returns:
            np.ndarray: 3D объем
        """
        if not images:
            return None
        
        # Создание 3D массива
        volume = np.stack(images, axis=0)
        
        # Нормализация размера по оси Z (количество срезов)
        target_slices = 64  # Стандартное количество срезов
        
        if volume.shape[0] > target_slices:
            # Уменьшение количества срезов
            indices = np.linspace(0, volume.shape[0] - 1, target_slices, dtype=int)
            volume = volume[indices]
        elif volume.shape[0] < target_slices:
            # Увеличение количества срезов (интерполяция)
            volume = cv2.resize(volume, (volume.shape[2], target_slices), interpolation=cv2.INTER_LINEAR)
            volume = np.transpose(volume, (1, 0, 2))
        
        return volume
