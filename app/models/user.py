from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import uuid
from pathlib import Path

from app import db

class Doctor(UserMixin, db.Model):
    """Модель врача для системы авторизации"""
    
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(200), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Связь с исследованиями
    studies = db.relationship('Study', backref='doctor', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, full_name, username, email, password):
        self.full_name = full_name
        self.username = username
        self.email = email
        self.set_password(password)
        self.create_user_directory()
    
    def set_password(self, password):
        """Хеширование пароля"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Проверка пароля"""
        return check_password_hash(self.password_hash, password)
    
    def create_user_directory(self):
        """Создание персональной папки врача в файловом хранилище"""
        user_dir = Path(f"data/studies/{self.username}")
        user_dir.mkdir(parents=True, exist_ok=True)
    
    def get_user_directory(self):
        """Получение пути к папке врача"""
        return Path(f"data/studies/{self.username}")
    
    def __repr__(self):
        return f'<Doctor {self.username}>'
