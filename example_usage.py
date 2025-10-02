#!/usr/bin/env python3
"""
Пример использования CTScanPredictor для предсказания патологий на КТ снимках
"""

import os
import sys
from pathlib import Path

# Добавляем путь к проекту в PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.utils.ct_predictor import CTScanPredictor


def main():
    """Пример использования предсказателя"""
    # Инициализация предсказателя
    predictor = CTScanPredictor()
    
    if predictor.model is None:
        print("Не удалось загрузить модель для предсказаний")
        return
    
    print("Модель успешно загружена!")
    print(f"Информация о модели: {predictor.get_model_info()}")
    
    # Пример 1: Предсказание для папки с исследованиями
    studies_folder = "./user_studies"  # Папка с исследованиями от пользователя
    
    if os.path.exists(studies_folder):
        print(f"Обработка папки с исследованиями: {studies_folder}")
        results = predictor.predict_from_folder(studies_folder)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
        print("="*60)
        
        for i, result in enumerate(results):
            print(f"\nИсследование {i+1}:")
            if result.get('status') == 'success':
                print(f"  Путь: {result['study_path']}")
                print(f"  Вероятность патологии: {result['probability_of_pathology']:.4f}")
                print(f"  Диагноз: {result['diagnosis']}")
                print(f"  Уверенность: {result['confidence']:.4f}")
                print(f"  Время обработки: {result['time_of_processing']:.2f} сек")
            else:
                print(f"  Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        
        print("="*60)
        
        # Сохраняем результаты в Excel
        predictor.save_predictions_to_excel("user_studies_prediction_results.xlsx")
        
    else:
        print(f"Папка с исследованиями не найдена: {studies_folder}")
        
    # Пример 2: Предсказание для одного исследования
    single_study_path = "./user_studies/study_001"  # Конкретное исследование
    
    if os.path.exists(single_study_path):
        print(f"\nОбработка отдельного исследования: {single_study_path}")
        result = predictor.predict_single_study(single_study_path)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ (ОДНО ИССЛЕДОВАНИЕ)")
        print("="*60)
        
        if result.get('status') == 'success':
            print(f"Путь: {result['study_path']}")
            print(f"Вероятность патологии: {result['probability_of_pathology']:.4f}")
            print(f"Диагноз: {result['diagnosis']}")
            print(f"Уверенность: {result['confidence']:.4f}")
            print(f"Время обработки: {result['time_of_processing']:.2f} сек")
        else:
            print(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        
        print("="*60)
        
        # Сохраняем результат в Excel
        predictor.save_predictions_to_excel("single_study_prediction_results.xlsx")
        
    else:
        print(f"Исследование не найдено: {single_study_path}")
    
    # Пример 3: Предсказание для ZIP архива
    zip_study_path = "./user_studies/ct_study.zip"  # ZIP архив с DICOM файлами
    
    if os.path.exists(zip_study_path):
        print(f"\nОбработка ZIP архива: {zip_study_path}")
        result = predictor.predict_single_study(zip_study_path)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ (ZIP АРХИВ)")
        print("="*60)
        
        if result.get('status') == 'success':
            print(f"Путь: {result['study_path']}")
            print(f"Вероятность патологии: {result['probability_of_pathology']:.4f}")
            print(f"Диагноз: {result['diagnosis']}")
            print(f"Уверенность: {result['confidence']:.4f}")
            print(f"Время обработки: {result['time_of_processing']:.2f} сек")
        else:
            print(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")
        
        print("="*60)
        
    else:
        print(f"ZIP архив не найден: {zip_study_path}")
    
    # Создание тестовых данных для демонстрации
    create_test_data()


def create_test_data():
    """Создание тестовых данных для демонстрации"""
    print("\n" + "="*60)
    print("СОЗДАНИЕ ТЕСТОВЫХ ДАННЫХ")
    print("="*60)
    
    # Создание папки для тестовых исследований
    test_folder = Path("./user_studies")
    test_folder.mkdir(exist_ok=True)
    
    print(f"Создана папка для тестовых данных: {test_folder}")
    print("Для полного тестирования поместите в эту папку:")
    print("1. ZIP архивы с DICOM файлами")
    print("2. Папки с DICOM файлами")
    print("\nПример структуры:")
    print("user_studies/")
    print("├── study_001/")
    print("│   ├── slice_001.dcm")
    print("│   ├── slice_002.dcm")
    print("│   └── ...")
    print("├── study_002.zip")
    print("└── study_003.zip")


if __name__ == '__main__':
    main()
