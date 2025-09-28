#!/usr/bin/env python3
"""
Скрипт для создания тестового DICOM файла
"""

import pydicom
import numpy as np
from pathlib import Path

def create_test_dicom():
    """Создает простой тестовый DICOM файл"""
    
    # Создаем простую 2D изображение (256x256 пикселей)
    pixel_array = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
    
    # Создаем DICOM dataset
    ds = pydicom.Dataset()
    
    # Добавляем file meta information
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    
    # Обязательные поля для DICOM
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    # Информация о пациенте
    ds.PatientName = "Test^Patient"
    ds.PatientID = "TEST001"
    ds.PatientBirthDate = "19900101"
    ds.PatientSex = "M"
    
    # Информация об исследовании
    ds.StudyDate = "20250101"
    ds.StudyTime = "120000"
    ds.StudyDescription = "Test CT Study"
    ds.StudyID = "TEST001"
    
    # Информация о серии
    ds.SeriesDate = "20250101"
    ds.SeriesTime = "120000"
    ds.SeriesDescription = "Test Series"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    
    # Информация об изображении
    ds.Modality = "CT"
    ds.Manufacturer = "Test Manufacturer"
    ds.ManufacturerModelName = "Test Model"
    ds.SoftwareVersions = "1.0"
    
    # Параметры изображения
    ds.Rows = 256
    ds.Columns = 256
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # Параметры CT
    ds.SliceThickness = 5.0
    ds.KVP = 120
    ds.ExposureTime = 1000
    ds.XRayTubeCurrent = 200
    ds.Exposure = 200
    
    # Параметры ориентации и позиции
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.SliceLocation = 0.0
    
    # Параметры пикселей
    ds.PixelSpacing = [1.0, 1.0]
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1.0
    
    # Добавляем пиксельные данные
    ds.PixelData = pixel_array.tobytes()
    
    # Создаем директорию для тестовых файлов
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Сохраняем файл с правильными параметрами
    filename = test_dir / "test_ct_001.dcm"
    ds.save_as(filename, write_like_original=False)
    
    print(f"Тестовый DICOM файл создан: {filename}")
    print(f"Размер файла: {filename.stat().st_size} байт")
    
    return filename

if __name__ == "__main__":
    create_test_dicom()
