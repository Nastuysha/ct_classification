from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path
from app import db
from app.models import Study
from app.utils.dicom_processor import DicomProcessor
from app.utils.ai_classifier import CTClassifier

studies_bp = Blueprint('studies', __name__)

@studies_bp.route('/')
@login_required
def list_studies():
    """Список всех исследований врача"""
    page = request.args.get('page', 1, type=int)
    studies = Study.query.filter_by(doctor_id=current_user.id)\
                       .order_by(Study.created_at.desc())\
                       .paginate(page=page, per_page=10, error_out=False)
    
    return render_template('studies/list.html', studies=studies)

@studies_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_study():
    """Загрузка нового исследования"""
    if request.method == 'POST':
        print(f"POST request received. Files: {request.files}")
        print(f"Form data: {request.form}")
        
        if 'dicom_files' not in request.files:
            print("No 'dicom_files' in request.files")
            flash('Файлы не выбраны', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('dicom_files')
        print(f"Files list: {[f.filename for f in files]}")
        
        if not files or files[0].filename == '':
            print("No files or empty filename")
            flash('Файлы не выбраны', 'error')
            return redirect(request.url)
        
        # Получение дополнительной информации
        patient_id = request.form.get('patient_id', '')
        study_description = request.form.get('study_description', '')
        
        try:
            # Создание нового исследования
            study = Study(
                doctor_id=current_user.id,
                patient_id=patient_id,
                study_description=study_description
            )
            
            db.session.add(study)
            db.session.commit()
            
            # Создание директории после сохранения в БД
            study.create_study_directory()
            
            # Сохранение DICOM файлов
            dicom_dir = study.get_dicom_directory()
            saved_files = []
            
            for file in files:
                print(f"Processing file: {file.filename}")
                if file and file.filename.lower().endswith(('.dcm', '.dicom')):
                    filename = secure_filename(file.filename)
                    file_path = dicom_dir / filename
                    print(f"Saving file to: {file_path}")
                    file.save(str(file_path))
                    saved_files.append(filename)
                    print(f"File saved successfully: {filename}")
                else:
                    print(f"File rejected (not DICOM): {file.filename}")
            
            print(f"Total saved files: {len(saved_files)}")
            
            if not saved_files:
                print("No valid DICOM files found")
                flash('Не найдено корректных DICOM файлов', 'error')
                db.session.delete(study)
                db.session.commit()
                return redirect(request.url)
            
            study.set_status('uploaded')
            db.session.commit()
            
            flash(f'Исследование {study.study_id} успешно загружено. Файлов: {len(saved_files)}', 'success')
            return redirect(url_for('studies.view_study', study_id=study.study_id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Ошибка при загрузке исследования: {str(e)}', 'error')
            print(f"Upload error: {e}")
    
    return render_template('studies/upload.html')

@studies_bp.route('/<study_id>')
@login_required
def view_study(study_id):
    """Просмотр конкретного исследования"""
    study = Study.query.filter_by(study_id=study_id, doctor_id=current_user.id).first_or_404()
    
    # Получение списка DICOM файлов
    dicom_files = []
    dicom_dir = study.get_dicom_directory()
    if dicom_dir.exists():
        dicom_files = [f.name for f in dicom_dir.iterdir() if f.is_file()]
    
    return render_template('studies/view.html', study=study, dicom_files=dicom_files)

@studies_bp.route('/<study_id>/classify', methods=['POST'])
@login_required
def classify_study(study_id):
    """Классификация исследования с помощью ИИ"""
    study = Study.query.filter_by(study_id=study_id, doctor_id=current_user.id).first_or_404()
    
    if study.status != 'uploaded':
        return jsonify({'error': 'Исследование уже обработано или находится в обработке'}), 400
    
    try:
        # Установка статуса "обработка"
        study.set_status('processing')
        db.session.commit()
        
        # Обработка DICOM файлов
        processor = DicomProcessor()
        processed_images = processor.process_study(study.get_dicom_directory())
        
        if not processed_images:
            study.set_status('error')
            db.session.commit()
            return jsonify({'error': 'Не удалось обработать DICOM файлы'}), 400
        
        # Классификация с помощью ИИ
        classifier = CTClassifier()
        result, confidence = classifier.classify_images(processed_images)
        
        # Сохранение результатов
        study.update_classification_result(result, confidence, classifier.get_model_version())
        db.session.commit()
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': confidence,
            'model_version': classifier.get_model_version()
        })
        
    except Exception as e:
        study.set_status('error')
        db.session.commit()
        print(f"Classification error: {e}")
        return jsonify({'error': f'Ошибка при классификации: {str(e)}'}), 500

@studies_bp.route('/<study_id>/delete', methods=['POST'])
@login_required
def delete_study(study_id):
    """Удаление исследования"""
    study = Study.query.filter_by(study_id=study_id, doctor_id=current_user.id).first_or_404()
    
    try:
        # Удаление файлов
        study_dir = study.get_study_directory()
        if study_dir.exists():
            import shutil
            shutil.rmtree(study_dir)
        
        # Удаление из БД
        db.session.delete(study)
        db.session.commit()
        
        flash('Исследование успешно удалено', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Ошибка при удалении исследования: {str(e)}', 'error')
        print(f"Delete error: {e}")
    
    return redirect(url_for('studies.list_studies'))
