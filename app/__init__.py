from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os
from pathlib import Path

# Инициализация расширений
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    
    # Конфигурация
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ct_classification.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = 'data/studies'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    
    # Инициализация расширений с приложением
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Пожалуйста, войдите в систему для доступа к этой странице.'
    
    # Настройка user_loader для Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import Doctor
        return Doctor.query.get(int(user_id))
    
    # Создание директорий
    create_directories()
    
    # Регистрация Blueprint'ов
    from app.views.auth import auth_bp
    from app.views.main import main_bp
    from app.views.studies import studies_bp
    from app.views.admin import admin_bp
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(studies_bp, url_prefix='/studies')
    app.register_blueprint(admin_bp)
    
    # Создание таблиц БД
    with app.app_context():
        db.create_all()
    
    return app

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        'data/studies',
        'data/models',
        'app/static/uploads',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
