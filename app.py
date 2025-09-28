#!/usr/bin/env python3
"""
КТ Классификатор ОГК - MVP приложение
Система автоматической классификации компьютерных томографических исследований 
органов грудной клетки с использованием искусственного интеллекта
"""

import os
import logging
from flask import Flask
from app import create_app, db
from app.models import Doctor, Study

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_sample_data():
    """Создание тестовых данных для демонстрации"""
    try:
        # Проверка существования тестового врача
        test_doctor = Doctor.query.filter_by(username='demo_doctor').first()
        
        if not test_doctor:
            # Создание тестового врача
            test_doctor = Doctor(
                full_name='Демо Врач Тестовый',
                username='demo_doctor',
                email='demo@hospital.ru',
                password='demo123'
            )
            
            db.session.add(test_doctor)
            db.session.commit()
            
            logger.info("Создан тестовый врач: demo_doctor / demo123")
            
            # Создание тестового исследования
            test_study = Study(
                doctor_id=test_doctor.id,
                patient_id='DEMO001',
                study_description='Демонстрационное исследование'
            )
            
            db.session.add(test_study)
            db.session.commit()
            
            # Создание директории после сохранения в БД
            test_study.create_study_directory()
            
            logger.info(f"Создано тестовое исследование: {test_study.study_id}")
        
    except Exception as e:
        logger.error(f"Ошибка при создании тестовых данных: {e}")

if __name__ == '__main__':
    # Создание приложения
    app = create_app()
    
    with app.app_context():
        # Создание таблиц БД
        db.create_all()
        logger.info("База данных инициализирована")
        
        # Создание тестовых данных
        create_sample_data()
    
    # Запуск приложения
    logger.info("Запуск приложения КТ Классификатор ОГК")
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
