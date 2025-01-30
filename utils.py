import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# 클래스 로딩 (예: COCO, Pascal VOC의 클래스)
def load_classes(file_name):
    with open(file_name, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Non-Maximum Suppression (중복된 바운딩 박스를 제거)
def non_max_suppression(prediction, conf_thresh=0.5, nms_thresh=0.4):
    # Prediction은 [batch_size, 7, 7, 30] 차원을 가지며,
    # 각 grid 셀에서 5개의 바운딩 박스와 20개의 클래스 확률을 예측함
    boxes = prediction[..., :4]  # 바운딩 박스 (x, y, w, h)
    confidences = prediction[..., 4]  # 신뢰도
    class_probs = prediction[..., 5:]  # 클래스 확률
    
    # Grid마다 박스를 나누고, 신뢰도 및 클래스별 확률 계산
    # (여기서 NMS 후처리를 통해 최종 바운딩 박스 선택)
    
    selected_boxes = []  # 이 곳에 최종 선택된 박스들 저장
    
    # 후처리 과정에 대한 기본적인 예시 구현:
    # confidence score가 일정 threshold 이상인 박스를 필터링
    for box, conf, class_prob in zip(boxes, confidences, class_probs):
        if conf > conf_thresh:
            selected_boxes.append(box)

    return selected_boxes

# 바운딩 박스를 이미지에 그리기
def plot_boxes(image, boxes):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # 각 바운딩 박스를 이미지 위에 표시 (여기서는 간단히 색상을 채택)
    for box in boxes:
        ax.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2], box[3], fill=False, color='red', linewidth=2))

    plt.show()