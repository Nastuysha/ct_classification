"""
CT Scan Predictor - Класс для предсказания патологий на КТ снимках
"""

import os
import time
import pickle
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import cv2
import logging

try:
    import tensorflow as tf
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow не установлен. Функции предсказания будут недоступны.")

logger = logging.getLogger(__name__)


class CTScanPredictor:
    """Класс для предсказания патологий на КТ снимках"""

    def __init__(self, model_path: Optional[str] = None, pipeline_path: Optional[str] = None):
        """
        Инициализация предсказателя

        Args:
            model_path: Путь к файлу модели (.keras)
            pipeline_path: Путь к файлу пайплайна (.pkl)
        """
        self.model = None
        self.pipeline = None
        self.model_version = "1.0.0"
        self.predictions_history = []

        # Пути по умолчанию
        if model_path is None:
            model_path = "data/models/ct_classification_model.keras"
        if pipeline_path is None:
            pipeline_path = "data/models/ct_classification_pipeline.pkl"

        self.model_path = model_path
        self.pipeline_path = pipeline_path

        # Загрузка модели и пайплайна
        self._load_model()
        self._load_pipeline()

    def _load_model(self):
        """Загрузка модели TensorFlow/Keras"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow не установлен")
            return

        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Модель успешно загружена из {self.model_path}")
            else:
                logger.error(f"Файл модели не найден: {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            self.model = None

    def _load_pipeline(self):
        """Загрузка пайплайна предобработки"""
        try:
            if os.path.exists(self.pipeline_path):
                with open(self.pipeline_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                logger.info(f"Пайплайн успешно загружен из {self.pipeline_path}")
            else:
                logger.warning(f"Файл пайплайна не найден: {self.pipeline_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке пайплайна: {e}")
            self.pipeline = None

    def _extract_zip_to_temp(self, zip_path: str) -> str:
        """
        Извлечение ZIP архива во временную папку

        Args:
            zip_path: Путь к ZIP файлу

        Returns:
            Путь к временной папке с извлеченными файлами
        """
        temp_dir = tempfile.mkdtemp()

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            logger.info(f"ZIP архив извлечен в {temp_dir}")
            return temp_dir
        except Exception as e:
            logger.error(f"Ошибка при извлечении ZIP архива: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _load_dicom_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """
        Загрузка и предобработка DICOM изображения

        Args:
            dicom_path: Путь к DICOM файлу

        Returns:
            Предобработанное изображение или None при ошибке
        """
        try:
            # Пропускаем системные файлы
            filename = Path(dicom_path).name
            if (filename.startswith('._') or
                    filename == '.DS_Store' or
                    '__MACOSX' in dicom_path):
                return None

            # Загрузка DICOM файла с force=True для проблемных файлов
            try:
                dicom_data = pydicom.dcmread(dicom_path)
            except Exception:
                # Попытка с force=True
                try:
                    dicom_data = pydicom.dcmread(dicom_path, force=True)
                except Exception:
                    logger.warning(f"Не удалось прочитать DICOM файл: {dicom_path}")
                    return None

            # Получение пиксельных данных
            # if hasattr(dicom_data, 'pixel_array'):
            #     try:
            #         # Попытка получить пиксельные данные
            #         image = dicom_data.pixel_array
            #     except Exception as e:
            #         # Попытка декомпрессии если данные сжаты
            #         try:
            #             dicom_data.decompress()
            #             image = dicom_data.pixel_array
            #         except Exception as e2:
            #             logger.warning(f"Не удалось получить пиксельные данные из {dicom_path}: {e}, {e2}")
            #             return None
            # el
            if hasattr(dicom_data, 'PixelData'):
                # Попытка работы с сырыми пиксельными данными
                try:
                    import numpy as np
                    pixel_data = dicom_data.PixelData
                    if pixel_data:
                        # Преобразование сырых данных в массив
                        rows = dicom_data.get('Rows', 512)
                        cols = dicom_data.get('Columns', 512)
                        bits_allocated = dicom_data.get('BitsAllocated', 16)

                        if bits_allocated == 16:
                            dtype = np.int16 if dicom_data.get('PixelRepresentation', 0) == 1 else np.uint16
                        else:
                            dtype = np.uint8

                        image = np.frombuffer(pixel_data, dtype=dtype).reshape(rows, cols)
                    else:
                        logger.warning(f"Пустые пиксельные данные: {dicom_path}")
                        return None
                except Exception as e:
                    logger.warning(f"Не удалось обработать сырые пиксельные данные {dicom_path}: {e}")
                    return None
            else:
                logger.warning(f"DICOM файл не содержит изображения: {dicom_path}")
                return None

            # Проверка размера изображения
            if image.size == 0:
                logger.warning(f"Пустое изображение: {dicom_path}")
                return None

            # Нормализация изображения
            if image.dtype != np.uint8:
                # Проверка на валидные значения
                if np.isnan(image).any() or np.isinf(image).any():
                    logger.warning(f"Изображение содержит NaN или Inf: {dicom_path}")
                    return None

                # Нормализация к диапазону 0-255
                if image.max() > image.min():
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                else:
                    image = np.zeros_like(image, dtype=np.uint8)

            # Изменение размера до стандартного (128x128 для данной модели)
            image = cv2.resize(image, (128, 128))

            # Оставляем как grayscale для медицинских изображений
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            return image

        except Exception as e:
            logger.error(f"Ошибка при загрузке DICOM файла {dicom_path}: {e}")
            return None

    def _preprocess_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Предобработка списка изображений для 3D модели

        Args:
            images: Список 2D изображений

        Returns:
            3D тензор для модели
        """
        if not images:
            return np.array([])

        # Ограничиваем количество срезов до 64 (как ожидает модель)
        max_slices = 64
        if len(images) > max_slices:
            # Берем равномерно распределенные срезы
            indices = np.linspace(0, len(images) - 1, max_slices, dtype=int)
            images = [images[i] for i in indices]
        elif len(images) < max_slices:
            # Дополняем нулевыми срезами
            while len(images) < max_slices:
                images.append(np.zeros((128, 128), dtype=np.uint8))

        # Нормализация изображений и создание 3D тензора
        volume = []
        for image in images:
            # Нормализация к диапазону [0, 1]
            normalized_image = image.astype(np.float32) / 255.0
            volume.append(normalized_image)

        # Создаем 3D тензор: (128, 128, 64, 1)
        volume = np.array(volume)  # (64, 128, 128)
        volume = np.transpose(volume, (1, 2, 0))  # (128, 128, 64)
        volume = np.expand_dims(volume, axis=-1)  # (128, 128, 64, 1)
        volume = np.expand_dims(volume, axis=0)  # (1, 128, 128, 64, 1)

        return volume

    def _get_dicom_files_from_folder(self, folder_path: str) -> List[str]:
        """
        Получение списка DICOM файлов из папки

        Args:
            folder_path: Путь к папке

        Returns:
            Список путей к DICOM файлам
        """
        dicom_files = []
        folder = Path(folder_path)

        if not folder.exists():
            return dicom_files

        # Поиск DICOM файлов
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                # Пропускаем системные файлы
                filename = file_path.name
                if (filename.startswith('._') or
                        filename == '.DS_Store' or
                        '__MACOSX' in str(file_path)):
                    continue

                # Проверка расширения
                if file_path.suffix.lower() in ['.dcm', '.dicom']:
                    dicom_files.append(str(file_path))
                else:
                    # Попытка определить DICOM файл по содержимому
                    try:
                        pydicom.dcmread(str(file_path), stop_before_pixels=True)
                        dicom_files.append(str(file_path))
                    except:
                        continue

        return sorted(dicom_files)

    def predict_single_study(self, study_path: str) -> Dict[str, Any]:
        """
        Предсказание для одного исследования

        Args:
            study_path: Путь к папке с исследованием или ZIP файлу

        Returns:
            Словарь с результатами предсказания
        """
        start_time = time.time()

        result = {
            'study_path': study_path,
            'status': 'error',
            'error': None,
            'probability_of_pathology': 0.0,
            'diagnosis': 'Неопределено',
            'confidence': 0.0,
            'time_of_processing': 0.0
        }

        if self.model is None:
            result['error'] = 'Модель не загружена'
            return result

        temp_dir = None
        try:
            # Определение типа входных данных
            if study_path.lower().endswith('.zip'):
                # Извлечение ZIP архива
                temp_dir = self._extract_zip_to_temp(study_path)
                dicom_files = self._get_dicom_files_from_folder(temp_dir)
            else:
                # Папка с DICOM файлами
                dicom_files = self._get_dicom_files_from_folder(study_path)

            if not dicom_files:
                result['error'] = 'DICOM файлы не найдены'
                return result

            # Загрузка и предобработка изображений
            images = []
            for dicom_file in dicom_files:
                image = self._load_dicom_image(dicom_file)
                if image is not None:
                    images.append(image)

            if not images:
                result['error'] = 'Не удалось загрузить изображения'
                return result

            # Предобработка для модели (создает 3D тензор)
            processed_volume = self._preprocess_images(images)

            if processed_volume.size == 0:
                result['error'] = 'Не удалось создать 3D тензор'
                return result

            # Предсказание для 3D тензора
            predictions = self.model.predict(processed_volume, verbose=0)

            # Получаем предсказание (уже для всего объема)
            avg_prediction = predictions[0]

            # Интерпретация результата
            if len(avg_prediction) == 1:
                # Бинарная классификация
                probability_pathology = float(avg_prediction[0])
            else:
                # Многоклассовая классификация - берем вероятность патологии
                probability_pathology = float(1 - avg_prediction[0])  # Предполагаем, что первый класс - норма

            # Определение диагноза
            if probability_pathology > 0.7:
                diagnosis = 'Патология обнаружена'
                confidence = probability_pathology
            elif probability_pathology > 0.3:
                diagnosis = 'Требуется дополнительное обследование'
                confidence = 0.5
            else:
                diagnosis = 'Патология не обнаружена'
                confidence = 1 - probability_pathology

            # Заполнение результата
            result.update({
                'status': 'success',
                'probability_of_pathology': probability_pathology,
                'diagnosis': diagnosis,
                'confidence': confidence,
                'time_of_processing': time.time() - start_time
            })

            # Сохранение в историю
            self.predictions_history.append(result.copy())

        except Exception as e:
            logger.error(f"Ошибка при предсказании для {study_path}: {e}")
            result['error'] = str(e)

        finally:
            # Очистка временной папки
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

            result['time_of_processing'] = time.time() - start_time

        return result

    def predict_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Предсказание для всех исследований в папке

        Args:
            folder_path: Путь к папке с исследованиями

        Returns:
            Список результатов предсказаний
        """
        results = []
        folder = Path(folder_path)

        if not folder.exists():
            logger.error(f"Папка не существует: {folder_path}")
            return results

        # Поиск ZIP файлов и папок с исследованиями
        for item in folder.iterdir():
            if item.is_file() and item.suffix.lower() == '.zip':
                # ZIP файл
                result = self.predict_single_study(str(item))
                results.append(result)
            elif item.is_dir():
                # Папка с исследованием
                dicom_files = self._get_dicom_files_from_folder(str(item))
                if dicom_files:
                    result = self.predict_single_study(str(item))
                    results.append(result)

        return results

    def save_predictions_to_excel(self, filename: str = "predictions_results.xlsx"):
        """
        Сохранение результатов предсказаний в Excel файл

        Args:
            filename: Имя файла для сохранения
        """
        if not self.predictions_history:
            logger.warning("Нет результатов для сохранения")
            return

        try:
            # Подготовка данных для DataFrame
            data = []
            for result in self.predictions_history:
                data.append({
                    'Путь к исследованию': result.get('study_path', ''),
                    'Статус': result.get('status', ''),
                    'Вероятность патологии': result.get('probability_of_pathology', 0.0),
                    'Диагноз': result.get('diagnosis', ''),
                    'Уверенность': result.get('confidence', 0.0),
                    'Время обработки (сек)': result.get('time_of_processing', 0.0),
                    'Ошибка': result.get('error', '')
                })

            # Создание DataFrame и сохранение в Excel
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False, engine='openpyxl')

            logger.info(f"Результаты сохранены в файл: {filename}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении в Excel: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели

        Returns:
            Словарь с информацией о модели
        """
        info = {
            'model_loaded': self.model is not None,
            'pipeline_loaded': self.pipeline is not None,
            'model_version': self.model_version,
            'model_path': self.model_path,
            'pipeline_path': self.pipeline_path,
            'predictions_count': len(self.predictions_history)
        }

        if self.model is not None:
            try:
                info['model_input_shape'] = self.model.input_shape
                info['model_output_shape'] = self.model.output_shape
            except:
                pass

        return info
