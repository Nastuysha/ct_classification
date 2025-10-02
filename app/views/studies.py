from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import uuid
import zipfile
import tempfile
import shutil
from pathlib import Path
from app import db
from app.models import Study
from app.utils.dicom_processor import DicomProcessor
from app.utils.ai_classifier import CTClassifier
from app.utils.ct_predictor import CTScanPredictor

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

        if 'zip_file' not in request.files:
            print("No 'zip_file' in request.files")
            flash('ZIP файл не выбран', 'error')
            return redirect(request.url)

        zip_file = request.files['zip_file']
        print(f"ZIP file: {zip_file.filename}")

        if not zip_file or zip_file.filename == '':
            print("No ZIP file or empty filename")
            flash('ZIP файл не выбран', 'error')
            return redirect(request.url)

        if not zip_file.filename.lower().endswith('.zip'):
            print("File is not a ZIP archive")
            flash('Файл должен быть ZIP архивом', 'error')
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

            # Сохранение ZIP файла и извлечение DICOM файлов
            dicom_dir = study.get_dicom_directory()
            zip_filename = secure_filename(zip_file.filename)
            zip_path = dicom_dir / zip_filename

            print(f"Saving ZIP file to: {zip_path}")
            zip_file.save(str(zip_path))

            # Извлечение DICOM файлов из ZIP архива
            extracted_files = []
            try:
                with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
                    for file_info in zip_ref.filelist:
                        if not file_info.is_dir():
                            # Пропускаем системные файлы macOS и другие служебные файлы
                            filename = file_info.filename
                            if (filename.startswith('__MACOSX/') or
                                filename.startswith('._') or
                                '._' in filename or
                                filename.endswith('.DS_Store')):
                                print(f"Skipping system file: {filename}")
                                continue

                            # Извлекаем файл
                            extracted_path = zip_ref.extract(file_info, str(dicom_dir))
                            extracted_files.append(extracted_path)
                            print(f"Extracted file: {extracted_path}")

                # Удаляем ZIP файл после извлечения
                os.remove(str(zip_path))

                print(f"Total extracted files: {len(extracted_files)}")

                if not extracted_files:
                    print("No files found in ZIP archive")
                    flash('ZIP архив пуст или поврежден', 'error')
                    db.session.delete(study)
                    db.session.commit()
                    return redirect(request.url)

            except zipfile.BadZipFile:
                print("Invalid ZIP file")
                flash('Поврежденный ZIP архив', 'error')
                db.session.delete(study)
                db.session.commit()
                return redirect(request.url)
            except Exception as e:
                print(f"Error extracting ZIP: {e}")
                flash(f'Ошибка при извлечении архива: {str(e)}', 'error')
                db.session.delete(study)
                db.session.commit()
                return redirect(request.url)

            study.set_status('uploaded')
            db.session.commit()

            flash(f'Исследование {study.study_id} успешно загружено. Файлов: {len(extracted_files)}', 'success')
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

    # Получение списка DICOM файлов (включая файлы в подпапках)
    dicom_files = []
    dicom_dir = study.get_dicom_directory()
    if dicom_dir.exists():
        # Рекурсивный поиск всех файлов в папке и подпапках
        for file_path in dicom_dir.rglob('*'):
            if file_path.is_file():
                # Показываем относительный путь от dicom_dir
                relative_path = file_path.relative_to(dicom_dir)
                dicom_files.append(str(relative_path))

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

        # Использование нового предсказателя
        predictor = CTScanPredictor()

        if predictor.model is None:
            study.set_status('error')
            db.session.commit()
            return jsonify({'error': 'Модель не загружена'}), 400

        # Предсказание для исследования
        prediction_result = predictor.predict_single_study(str(study.get_dicom_directory()))

        if prediction_result['status'] != 'success':
            study.set_status('error')
            db.session.commit()
            return jsonify({'error': prediction_result.get('error', 'Ошибка при предсказании')}), 400

        # Сохранение результатов
        diagnosis = prediction_result['diagnosis']
        confidence = prediction_result['confidence']
        probability_pathology = prediction_result['probability_of_pathology']
        model_version = predictor.model_version

        # Преобразуем диагноз в формат, ожидаемый шаблоном
        if 'не обнаружена' in diagnosis.lower() or 'норма' in diagnosis.lower():
            result = 'normal'
        elif 'обнаружена' in diagnosis.lower() or 'патология' in diagnosis.lower():
            result = 'pathology'
        else:
            result = 'uncertain'

        study.update_classification_result(result, confidence, model_version)

        # Сохраняем дополнительную информацию
        study.diagnosis_text = diagnosis
        study.probability_pathology = probability_pathology
        db.session.commit()

        # Сохранение результатов в Excel (история) и Полный отчет
        try:
            results_dir = study.get_results_directory()
            excel_filename = f"study_{study.study_id}_results.xlsx"
            excel_path = results_dir / excel_filename
            predictor.save_predictions_to_excel(str(excel_path))

            # Полный отчет с 7 колонками
            full_report_filename = "full_report.xlsx"
            full_report_path = results_dir / full_report_filename
            predictor.save_full_report_to_excel(str(full_report_path), prediction_result)
        except Exception as e:
            print(f"Error saving Excel: {e}")

        return jsonify({
            'success': True,
            'result': result,
            'confidence': confidence,
            'probability_of_pathology': prediction_result['probability_of_pathology'],
            'model_version': model_version,
            'processing_time': prediction_result['time_of_processing']
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

@studies_bp.route('/<study_id>/report', methods=['GET'])
@login_required
def download_full_report(study_id):
    """Скачивание полного отчета (Excel) для исследования"""
    study = Study.query.filter_by(study_id=study_id, doctor_id=current_user.id).first_or_404()
    full_report_path = study.get_results_directory() / 'full_report.xlsx'
    if not full_report_path.exists():
        flash('Полный отчет не найден. Повторите анализ исследования.', 'error')
        return redirect(url_for('studies.view_study', study_id=study.study_id))
    return send_file(str(full_report_path), as_attachment=True, download_name=f"full_report_{study.study_id}.xlsx")
