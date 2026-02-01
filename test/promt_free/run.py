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
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val_pe_free import YOLOEPEFreeDetectValidator
import supervision as sv
from simple_colors import green

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





unfused_model = YOLOE("yoloe-11l.yaml")
unfused_model.load('/mnt/nvme/huggingface/models--jameslahm--yoloe/snapshots/main/yoloe-11l-seg.pt')
unfused_model.eval()
unfused_model.cpu()

with open('yoloefork/tools/ram_tag_list.txt', 'r') as f:
    names = [x.strip() for x in f.readlines()]
with torch.no_grad():
    vocab = unfused_model.get_vocab(
        names,
        ckptfile='/mnt/nvme/huggingface/models--jameslahm--yoloe/snapshots/main/mobileclip_blt.pt'
    )


print(green('vocal ready'))
model = YOLOE("/mnt/nvme/huggingface/models--jameslahm--yoloe/snapshots/main/yoloe-11l-seg.pt").cuda()
model.set_vocab(vocab, names=names)
model.model.model[-1].is_fused = True
model.model.model[-1].conf = 0.001
model.model.model[-1].max_det = 1000


# model.predict('ultralytics/assets/bus.jpg', save=True)

