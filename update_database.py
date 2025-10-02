#!/usr/bin/env python3
"""
Скрипт для обновления базы данных с новыми полями
"""

import os
import sys
from pathlib import Path

# Добавляем путь к проекту в PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import create_app, db

def update_database():
    """Обновление базы данных с новыми полями"""
    app = create_app()
    
    with app.app_context():
        try:
            # Добавляем новые колонки в таблицу studies
            with db.engine.connect() as conn:
                conn.execute(db.text("""
                    ALTER TABLE studies 
                    ADD COLUMN diagnosis_text TEXT;
                """))
                conn.commit()
            print("Добавлена колонка diagnosis_text")
        except Exception as e:
            print(f"Колонка diagnosis_text уже существует или ошибка: {e}")
        
        try:
            with db.engine.connect() as conn:
                conn.execute(db.text("""
                    ALTER TABLE studies 
                    ADD COLUMN probability_pathology FLOAT;
                """))
                conn.commit()
            print("Добавлена колонка probability_pathology")
        except Exception as e:
            print(f"Колонка probability_pathology уже существует или ошибка: {e}")
        
        print("База данных обновлена!")

if __name__ == '__main__':
    update_database()
