from ultralytics import YOLOE
import cv2
import os
import re
from datetime import datetime
from pytz import timezone  # Добавь import
import torch
from torch import nn
from ultralytics.utils.torch_utils import smart_inference_mode
import mobileclip
from ultralytics import MobileCLIP
from time import time_ns
import torch

import supervision as sv
# Твои функции (исправлена ошибка)
def get_timestamp(filename):
    match = re.search(r'__CAM_FRONT__(\d{16})\.jpg$', filename)
    return int(match.group(1)) if match else 0

# Функция sweep'а - непрерывная последовательность кадров
def get_sweeps(jpgspaths, timestamps):
    sweeps = []
    for i in range(len(jpgspaths)):
        if i == 0 or timestamps[i] - timestamps[i-1] > 100000000:  # Новый sweep при разрыве >100ms
            sweeps.append({'start_idx': i, 'frames': []})
        sweeps[-1]['frames'].append(jpgspaths[i])
    return [s for s in sweeps if len(s['frames']) >= 5]  # Только sweeps с >=5 кадрами

# Функция кадров внутри sweep'а
def frames(sweep):
    return sweep['frames']


if __name__ == "__main__":
    # prompt = ["car","person"]
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
    model.set_classes(prompt, model.get_text_pe(prompt,ckptfile))


    jpgsroot = '/mnt/nvme/rowdata/nu/sweeps/CAM_FRONT'
    jpgspaths = [os.path.join(jpgsroot, el) for el in os.listdir(jpgsroot) if el.endswith('.jpg')]
    jpgspaths.sort(key=lambda path: get_timestamp(os.path.basename(path)))
    timestamps = [get_timestamp(os.path.basename(p)) for p in jpgspaths]
    sweeps = get_sweeps(jpgspaths, timestamps)
    print(f"Найдено {len(sweeps)} sweeps")

    window_name = f'jpgsroot'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # for sweep in sweeps:
    for sweep_idx, sweep in enumerate(sweeps):
        print(f"\n=== Sweep {sweep_idx+1}: {len(sweep['frames'])} кадров ===")
        
        # Находим индексы кадров в основном списке для доступа к timestamps
        frame_indices = [jpgspaths.index(path) for path in sweep['frames']]
        
        for frame_idx, path in enumerate(frames(sweep)):

            global_idx = frame_indices[frame_idx]
            image = cv2.imread(path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB для модели
            
            print(f'image size {image_rgb.shape}')
            torch.cuda.synchronize()
            t1 = time_ns()
            with torch.no_grad():
                results = model.predict(image_rgb, verbose=False)
            torch.cuda.synchronize()
            t2 = time_ns()
            print(f"eval time {(t2-t1)/1e6:.2f} ms")
            
            # ← КРИТИЧНО: получаем имена классов из модели
            class_names = model.names
            
            detections = sv.Detections.from_ultralytics(results[0])
            
            # ← ИСПРАВЛЕНО: правильный способ получить labels
            labels = [
                # f"{class_names[int(cl_id)]} {conf:.2f}"
                f"{class_names[int(cl_id)]}"
                for cl_id, conf in zip(detections.class_id, detections.confidence)
            ]
            
            resolution_wh = image_rgb.shape[:2]  # (H, W) - исправлено!
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
            
            # Заголовок sweep/frame
            title = f"Sweep {sweep_idx+1}/{len(sweeps)} | Frame {frame_idx+1}/{len(sweep['frames'])}"
            
            annotated_image = image_rgb.copy()
            annotated_image = sv.MaskAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                opacity=0.4
            ).annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                thickness=thickness
            ).annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                text_scale=text_scale,
                smart_position=True
            ).annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # ← Обратно в BGR для cv2.imshow
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, annotated_image_bgr)
    
        
            # Вычисляем реальную паузу между кадрами (в микросекундах)
            if frame_idx == 0:
                delay_us = 33000  # ~33ms для первого кадра (стандартный FPS камеры)
            else:
                prev_global_idx = frame_indices[frame_idx-1]
                delay_us = timestamps[global_idx] - timestamps[prev_global_idx]
            
            # Конвертируем микросекунды в миллисекунды для waitKey (с ограничением)
            delay_ms = min(delay_us // 1000, 1000)  # max 1 сек
            
            print(f"Frame {frame_idx+1}: delay={delay_us//1000}ms")
            
            key = cv2.waitKey(delay_ms)
            if key == 27 or key == ord('q'):  # ESC или Q
                cv2.destroyAllWindows()
                exit()
            if key == ord('n'):  # N - следующий sweep
                break
        else:
            print("Sweep завершен")
            key = cv2.waitKey(1000)  # Пауза 1с между sweeps
            continue
        break  # Выход из внешнего цикла при 'n'

    cv2.destroyAllWindows()
