import os
import sys
import cv2
import copy
import numpy as np
from time import time as time_module
import onnx
import onnxruntime as ort
import torch

# Загрузка ONNX-модели
onnx_model_path = 'weights/best.onnx'

# Установка сессии ONNX Runtime
session = ort.InferenceSession(onnx_model_path)

# Размер входного изображения для модели YOLOv9
FUSED_SHAPE = (1280, 1280)

def preprocess_image(image, input_shape):
    """Подготовка изображения для модели ONNX YOLOv9"""
    resized_image = cv2.resize(image, input_shape)
    normalized_image = resized_image / 255.0  # нормализация
    input_image = normalized_image.transpose(2, 0, 1).astype(np.float32)  # Приведение к формату NCHW
    input_image = np.expand_dims(input_image, axis=0)  # Добавляем batch dimension
    return input_image

def postprocess_output(output, input_shape):
    """Постобработка выхода модели ONNX"""
    predictions = output[0].reshape(-1, 9)  # Пример для YOLOv9 с 9 выходными параметрами на каждую рамку
    # Здесь можно добавить обработку, такую как фильтрация рамок по уверенности и классу
    # Пока что выводим просто "сырые" рамки
    return predictions

def process_frame(frame, session, input_shape=(1280, 640)):
    """Обработка одного кадра с использованием модели ONNX"""
    input_image = preprocess_image(frame, input_shape)
    ort_inputs = {session.get_inputs()[0].name: input_image}
    
    # Выполняем инференс
    output = session.run(None, ort_inputs)
    
    # Постобработка
    predictions = postprocess_output(output, input_shape)
    
    # Пример визуализации
    annotated_frame = frame.copy()
    for pred in predictions:
        x_center, y_center, width, height = pred[:4]
        xmin = int((x_center - width / 2) * frame.shape[1])
        ymin = int((y_center - height / 2) * frame.shape[0])
        xmax = int((x_center + width / 2) * frame.shape[1])
        ymax = int((y_center + height / 2) * frame.shape[0])
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return annotated_frame

def calculate_fps(start_time, num_frames):
    """Расчет FPS"""
    return num_frames / (time_module() - start_time)


if __name__ == "__main__":
    
    # Задаем путь к видео
    video_path = 'videos/track.mp4'
    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened() == False): 
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Размер видео: {frame_width}x{frame_height}, FPS: {fps}, Количество кадров: {total_frames}")

    output_video_path = 'output_track.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    start_time = time_module()
    num_frames = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        frame_count += 1
        num_frames += 1
        current_fps = calculate_fps(start_time, num_frames)
        
        if frame_count % 5 == 0 or frame_count == 1:
            if ret == True:
                # Инференс кадра через ONNX модель
                annotated_frame = process_frame(frame, session, FUSED_SHAPE)
                cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Frame', annotated_frame)
            
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        else:
            cv2.imshow('Frame', annotated_frame)    
        
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
