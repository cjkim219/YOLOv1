import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import config as con
from torchvision import transforms
from torchvision.datasets import VOCDetection
from PIL import Image
from models import YOLOv1  # YOLOv1 모델 구현 (자체 구현한 모델)
from utils import load_classes, non_max_suppression, plot_boxes  # 후처리와 시각화 유틸
   
# Train Dataset
# train_data = VOCDetection(root=con.root, year=con.year, image_set=con.image_set_train, download=True)

# Validation Dataset
# val_data = VOCDetection(root=con.root, year=con.year, image_set=con.image_set_val, download=True)

# TrainVal Dataset
# trainval_data = VOCDetection(root=con.root, year=con.year, image_set=con.image_set_trainval, download=True)



# 테스트할 이미지를 로드하고 전처리
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((448, 448)),  # YOLO v1의 입력 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 배치 크기 추가

# 모델 예측 후 결과를 후처리
def predict(model, image):
    model.eval()  # 평가 모드로 전환
    with torch.no_grad():
        pred = model(image)  # 모델 추론
    return pred

# 이미지에 예측된 바운딩 박스 출력
def display_predictions(image_path, prediction):
    image = Image.open(image_path)
    boxes, confidences, class_probs = non_max_suppression(prediction)
    plot_boxes(image, boxes)  # 시각화 유틸을 사용해서 이미지에 바운딩 박스 그리기
    plt.show()

if __name__ == "__main__":
    # 모델 인스턴스 생성 (사전 학습된 모델 로드 필요)
    model = YOLOv1()
    model.load_state_dict(torch.load("yolov1_weights.pth"))  # 사전 훈련된 모델 로딩 (예: pascal VOC)

    image_path = "test_image.jpg"  # 테스트할 이미지 경로
    image = prepare_image(image_path)  # 이미지 전처리

    # 이미지 예측
    prediction = predict(model, image)

    # 예측 결과를 이미지에 표시
    display_predictions(image_path, prediction)