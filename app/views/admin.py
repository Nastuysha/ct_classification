from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models.user import Doctor
from app.models.study import Study
import sqlite3
from pathlib import Path

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin/database')
@login_required
def view_database():
    """Просмотр базы данных (только для демо)"""

    # Получаем путь к базе данных
    db_path = Path("instance/ct_classification.db")

    if not db_path.exists():
        flash('База данных не найдена', 'error')
        return redirect(url_for('main.dashboard'))

    try:
        # Подключаемся к базе данных
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
        cursor = conn.cursor()

        # Получаем информацию о таблицах
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # Получаем данные из таблицы doctors
        cursor.execute("SELECT * FROM doctors")
        doctors_data = [dict(row) for row in cursor.fetchall()]

        # Получаем данные из таблицы studies
        cursor.execute("SELECT * FROM studies")
        studies_data = [dict(row) for row in cursor.fetchall()]

        # Получаем схему таблиц
        schemas = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schemas[table] = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return render_template('admin/database.html',
                             tables=tables,
                             doctors_data=doctors_data,
                             studies_data=studies_data,
                             schemas=schemas)

    except Exception as e:
        flash(f'Ошибка при чтении базы данных: {str(e)}', 'error')
        return redirect(url_for('main.dashboard'))

@admin_bp.route('/admin/query')
@login_required
def execute_query():
    """Выполнение SQL запросов (только для демо)"""

    query = request.args.get('q', '')
    if not query:
        return render_template('admin/query.html')

    db_path = Path("instance/ct_classification.db")

    if not db_path.exists():
        flash('База данных не найдена', 'error')
        return redirect(url_for('admin.view_database'))

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Выполняем запрос
        cursor.execute(query)

        # Определяем тип запроса
        query_upper = query.upper().strip()

        if query_upper.startswith('SELECT'):
            results = [dict(row) for row in cursor.fetchall()]
            columns = [description[0] for description in cursor.description] if cursor.description else []
            return render_template('admin/query.html',
                                 query=query,
                                 results=results,
                                 columns=columns)
        else:
            # Для INSERT, UPDATE, DELETE
            conn.commit()
            flash(f'Запрос выполнен успешно. Затронуто строк: {cursor.rowcount}', 'success')
            return redirect(url_for('admin.execute_query'))

    except Exception as e:
        flash(f'Ошибка при выполнении запроса: {str(e)}', 'error')
        return render_template('admin/query.html', query=query)

    finally:
        if 'conn' in locals():
            conn.close()
