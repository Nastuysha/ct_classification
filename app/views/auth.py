from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from app import db
from app.models import Doctor

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Регистрация нового врача"""
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Валидация
        if not all([full_name, username, email, password, confirm_password]):
            flash('Все поля обязательны для заполнения', 'error')
            return render_template('auth/register.html')
        
        if password != confirm_password:
            flash('Пароли не совпадают', 'error')
            return render_template('auth/register.html')
        
        if len(password) < 6:
            flash('Пароль должен содержать минимум 6 символов', 'error')
            return render_template('auth/register.html')
        
        # Проверка существования пользователя
        if Doctor.query.filter_by(username=username).first():
            flash('Пользователь с таким логином уже существует', 'error')
            return render_template('auth/register.html')
        
        if Doctor.query.filter_by(email=email).first():
            flash('Пользователь с таким email уже существует', 'error')
            return render_template('auth/register.html')
        
        try:
            # Создание нового врача
            doctor = Doctor(
                full_name=full_name,
                username=username,
                email=email,
                password=password
            )
            
            db.session.add(doctor)
            db.session.commit()
            
            flash('Регистрация прошла успешно! Теперь вы можете войти в систему.', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            flash('Ошибка при регистрации. Попробуйте еще раз.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('auth/register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Вход в систему"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))
        
        if not username or not password:
            flash('Введите логин и пароль', 'error')
            return render_template('auth/login.html')
        
        doctor = Doctor.query.filter_by(username=username).first()
        
        if doctor and doctor.check_password(password):
            if doctor.is_active:
                login_user(doctor, remember=remember)
                flash(f'Добро пожаловать, {doctor.full_name}!', 'success')
                return redirect(url_for('main.dashboard'))
            else:
                flash('Ваш аккаунт деактивирован. Обратитесь к администратору.', 'error')
        else:
            flash('Неверный логин или пароль', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Выход из системы"""
    logout_user()
    flash('Вы успешно вышли из системы', 'info')
    return redirect(url_for('auth.login'))
