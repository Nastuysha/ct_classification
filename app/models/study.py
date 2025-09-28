from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
import os
from pathlib import Path

from app import db

class Study(db.Model):
    """Модель исследования КТ ОГК"""
    
    __tablename__ = 'studies'
    
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    
    # Метаданные исследования
    patient_id = db.Column(db.String(100))
    study_date = db.Column(db.DateTime)
    study_description = db.Column(db.Text)
    
    # Результаты классификации
    classification_result = db.Column(db.String(50))  # 'normal' или 'pathology'
    confidence_score = db.Column(db.Float)
    model_version = db.Column(db.String(50))
    
    # Временные метки
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    
    # Статус обработки
    status = db.Column(db.String(20), default='uploaded')  # uploaded, processing, completed, error
    
    def __init__(self, doctor_id, patient_id=None, study_description=None):
        self.doctor_id = doctor_id
        self.patient_id = patient_id
        self.study_description = study_description
        self.study_id = str(uuid.uuid4())
        # Создание директории будет выполнено после сохранения в БД
    
    def create_study_directory(self):
        """Создание папки для конкретного исследования"""
        doctor = self.doctor
        study_dir = doctor.get_user_directory() / self.study_id
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание подпапок
        (study_dir / 'dicom').mkdir(exist_ok=True)
        (study_dir / 'processed').mkdir(exist_ok=True)
        (study_dir / 'results').mkdir(exist_ok=True)
    
    def get_study_directory(self):
        """Получение пути к папке исследования"""
        doctor = self.doctor
        return doctor.get_user_directory() / self.study_id
    
    def get_dicom_directory(self):
        """Получение пути к папке с DICOM файлами"""
        return self.get_study_directory() / 'dicom'
    
    def get_processed_directory(self):
        """Получение пути к папке с обработанными изображениями"""
        return self.get_study_directory() / 'processed'
    
    def get_results_directory(self):
        """Получение пути к папке с результатами"""
        return self.get_study_directory() / 'results'
    
    def update_classification_result(self, result, confidence, model_version):
        """Обновление результатов классификации"""
        self.classification_result = result
        self.confidence_score = confidence
        self.model_version = model_version
        self.processed_at = datetime.utcnow()
        self.status = 'completed'
    
    def set_status(self, status):
        """Обновление статуса исследования"""
        self.status = status
        if status == 'completed':
            self.processed_at = datetime.utcnow()
    
    def __repr__(self):
        return f'<Study {self.study_id}>'
