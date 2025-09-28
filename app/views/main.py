from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from app.models import Study

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Главная страница"""
    try:
        if current_user.is_authenticated:
            return redirect(url_for('main.dashboard'))
    except:
        pass  # Если пользователь не загружен, просто показываем главную страницу
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Панель управления врача"""
    # Получение последних исследований врача
    recent_studies = Study.query.filter_by(doctor_id=current_user.id)\
                              .order_by(Study.created_at.desc())\
                              .limit(10).all()
    
    # Статистика
    total_studies = Study.query.filter_by(doctor_id=current_user.id).count()
    completed_studies = Study.query.filter_by(doctor_id=current_user.id, status='completed').count()
    processing_studies = Study.query.filter_by(doctor_id=current_user.id, status='processing').count()
    
    stats = {
        'total': total_studies,
        'completed': completed_studies,
        'processing': processing_studies,
        'normal': Study.query.filter_by(doctor_id=current_user.id, classification_result='normal').count(),
        'pathology': Study.query.filter_by(doctor_id=current_user.id, classification_result='pathology').count()
    }
    
    return render_template('dashboard.html', 
                         recent_studies=recent_studies, 
                         stats=stats)
