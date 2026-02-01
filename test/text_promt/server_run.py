from ultralytics import YOLOE
import cv2
import os
import re
from datetime import datetime
from pytz import timezone
import torch
from torch import nn
from ultralytics.utils.torch_utils import smart_inference_mode
import mobileclip
from ultralytics import MobileCLIP
from time import time_ns
import torch
import supervision as sv
import numpy as np
from tqdm import tqdm 


# Твои функции
def get_timestamp(filename):
    match = re.search(r'__CAM_FRONT__(\d{16})\.jpg$', filename)
    return int(match.group(1)) if match else 0


def get_sweeps(jpgspaths, timestamps):
    sweeps = []
    for i in range(len(jpgspaths)):
        if i == 0 or timestamps[i] - timestamps[i-1] > 100000000:  # Новый sweep при разрыве >100ms
            sweeps.append({'start_idx': i, 'frames': []})
        sweeps[-1]['frames'].append(jpgspaths[i])
    return [s for s in sweeps if len(s['frames']) >= 5]  # Только sweeps с >=5 кадрами


def frames(sweep):
    return sweep['frames']


if __name__ == "__main__":
    # Настройки
    MAX_SWEEPS = 10  # Количество sweeps для записи (0 = все)
    OUTPUT_DIR = '/mnt/nvme/tmp_output_videos/promt_detection'
    # OUTPUT_DIR = '/mnt/nvme/tmp_output_videos/promt_road'
    FPS = 10  # FPS выходного видео
    
    # Создаем папку для видео если не существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    prompt = [
        # 🚗 VEHICLES (все типы)
        "car", "truck", "bus", "articulated-bus", "school-bus", "tour-bus", 
        "box-truck", "flatbed-truck", "dump-truck", "tanker-truck",
        "delivery-truck", "garbage-truck", "fire-truck", "ambulance", 
        "police-car", "taxi", "van", "minivan", "pickup", "suv",
        
        # 🏍️ TWO-WHEELERS
        "motorcycle", "moped", "scooter", "bicycle", "e-bike", "e-scooter",
        
        # 🚶 PEDESTRIANS (все подтипы)
        "pedestrian", "person", "child", "adult", "senior", "police", 
        "firefighter", "construction-worker", "delivery-person",
        "person-sitting", "person-bending", "person-on-phone",
        
        # 🐕 ANIMALS
        "dog", "cat", "bird", "squirrel", "raccoon", "deer", "coyote", 
        "goat", "pig",
    
        # 🚦 TRAFFIC CONTROL
        "traffic-light", "traffic-light-left", "traffic-light-right", 
        "sign", "dynamic-sign",
        
        # 🏗️ WORK-ZONES
        "construction-cone", "construction-barrel", "construction-barrier", 
        "jersey-barrier", "construction-fence", "construction-sign", 
        "excavator", "backhoe", "crane", "forklift",
        
        # 🛑 ROAD HAZARDS
        "pothole", "fallen-tree", "debris", "broken-glass", "oil-spill",
        
        # 🏪 URBAN
        "fire-hydrant", "parking-meter", "mailbox", "trash-can", "bench", 
        "bike-rack",
        
        # 🏛️ POLES
        "pole", "traffic-pole", "street-light", "light-pole", 
        "sign-pole", "utility-pole", "bollard",
        
        # 🏠 STRUCTURES
        "bridge", "tunnel", "overpass"
    ]
    # prompt = ["road"]

    device = 'cuda:0'
    model = YOLOE(
        model='/mnt/nvme/huggingface/models--jameslahm--yoloe/snapshots/main/yoloe-11l-seg.pt',
        task='segment',
        verbose=False
    )
    model.to(device)
    model.eval()


    ckptfile='/mnt/nvme/huggingface/models--jameslahm--yoloe/snapshots/main/mobileclip_blt.pt'
    model.set_classes(prompt, model.get_text_pe(prompt, ckptfile))


    jpgsroot = '/mnt/nvme/rowdata/nu/sweeps/CAM_FRONT'
    jpgspaths = [os.path.join(jpgsroot, el) for el in os.listdir(jpgsroot) if el.endswith('.jpg')]
    jpgspaths.sort(key=lambda path: get_timestamp(os.path.basename(path)))
    timestamps = [get_timestamp(os.path.basename(p)) for p in jpgspaths]
    sweeps = get_sweeps(jpgspaths, timestamps)
    
    print(f"Найдено {len(sweeps)} sweeps")
    print(f"Будет обработано sweeps: {min(MAX_SWEEPS, len(sweeps)) if MAX_SWEEPS > 0 else 'все'}")
    
    # Настраиваем аннотаторы один раз
    mask_annotator = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4
    )
    box_annotator = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=2  # Фиксированная толщина для видео
    )
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=0.5,  # Фиксированный размер текста
        text_thickness=1,
        smart_position=True
    )
    
    total_frames = 0
    total_inference_time = 0
    
    # Список для хранения путей к отдельным видео sweeps
    individual_videos = []
    
    # Обрабатываем sweeps
    for sweep_idx, sweep in enumerate(sweeps[:MAX_SWEEPS]):
        print(f"\n=== Обработка Sweep {sweep_idx+1}/{min(MAX_SWEEPS, len(sweeps))}: {len(sweep['frames'])} кадров ===")
        
        frame_indices = [jpgspaths.index(path) for path in sweep['frames']]
        
        # Создаем видео writer для текущего sweep
        first_frame = cv2.imread(frames(sweep)[0])
        if first_frame is None:
            continue
            
        h, w = first_frame.shape[:2]
        video_path = os.path.join(OUTPUT_DIR, f'sweep_{sweep_idx+1:03d}_{len(sweep["frames"])}frames.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
        
        print(f"Запись в: {video_path}")
        
        frame_count = 0
        for frame_idx, path in enumerate(frames(sweep)):
            image = cv2.imread(path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            torch.cuda.synchronize()
            t1 = time_ns()
            with torch.no_grad():
                results = model.predict(image_rgb, verbose=False)
            torch.cuda.synchronize()
            t2 = time_ns()
            inference_time = (t2 - t1) / 1e6
            total_inference_time += inference_time
            total_frames += 1
            
            class_names = model.names
            detections = sv.Detections.from_ultralytics(results[0])
            
            labels = [
                f"{class_names[int(cl_id)]}"
                for cl_id, conf in zip(detections.class_id, detections.confidence)
            ]
            
            # Аннотируем кадр
            annotated_image = image_rgb.copy()
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # BGR для записи
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Записываем кадр
            out.write(annotated_image_bgr)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Записано {frame_count}/{len(sweep['frames'])} кадров")
        
        out.release()
        print(f"  Sweep {sweep_idx+1} завершен: {frame_count} кадров записано")
        individual_videos.append(video_path)  # Добавляем путь к видео
    
    # === СОЗДАНИЕ ОБЪЕДИНЕННОГО ВИДЕО ===
    print(f"\n=== Создание объединенного видео из {len(individual_videos)} файлов ===")
    
    # Берем размеры первого видео
    cap = cv2.VideoCapture(individual_videos[0])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_combined = FPS  # Тот же FPS
    cap.release()
    
    combined_video_path = os.path.join(OUTPUT_DIR, f'combined_all_sweeps_{len(individual_videos)}_sweeps.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    combined_out = cv2.VideoWriter(combined_video_path, fourcc, fps_combined, (w, h))
    
    total_combined_frames = 0
    for video_idx, video_path in enumerate(individual_videos):
        print(f"  Добавление видео {video_idx+1}/{len(individual_videos)}: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            combined_out.write(frame)
            frame_count += 1
            total_combined_frames += 1
            
            if frame_count % 100 == 0:
                print(f"    Добавлено {frame_count} кадров из текущего видео")
        
        cap.release()
        print(f"    Видео {video_idx+1} добавлено: {frame_count} кадров")
    
    combined_out.release()
    print(f"Объединенное видео сохранено: {combined_video_path}")
    print(f"Всего кадров в объединенном видео: {total_combined_frames}")
    
    print(f"\n=== СТАТИСТИКА ===")
    print(f"Обработано sweeps: {min(MAX_SWEEPS, len(sweeps))}")
    print(f"Всего кадров: {total_frames}")
    print(f"Среднее время инференса: {total_inference_time/total_frames:.2f} ms/кадр")
    print(f"Индивидуальные видео сохранены в: {OUTPUT_DIR}")
    print(f"Объединенное видео: {combined_video_path}")
