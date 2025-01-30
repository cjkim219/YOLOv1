import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 이미지 크기 (448x448)에서 첫 번째 Conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 128, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv12 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv16 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv18 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv20 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)      
        self.conv21 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)   
        self.conv23 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv24 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)             
        
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)  # YOLO v1의 출력 크기
        self.fc2 = nn.Linear(4096, 7 * 7 * 30)
        
        self.dropout = nn.Dropout()
        self.activation = nn.LeakyReLU()
        self.batchNorm64 = nn.BatchNorm2d(64)
        self.batchNorm128 = nn.BatchNorm2d(128)
        self.batchNorm192 = nn.BatchNorm2d(192)
        self.batchNorm256 = nn.BatchNorm2d(256)
        self.batchNorm512 = nn.BatchNorm2d(512)
        self.batchNorm1024 = nn.BatchNorm2d(1024)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        

    def forward(self, x):
        
        # conv Layer-1
        x = self.activation(self.batchNorm64(self.conv1(x)))
        x = self.maxpool(x)
        
        # conv Layer-2
        x = self.activation(self.batchNorm192(self.conv2(x)))
        x = self.maxpool(x)
        
        # conv Layer-3
        x = self.activation(self.batchNorm128(self.conv3(x)))
        x = self.activation(self.batchNorm256(self.conv4(x)))
        x = self.activation(self.batchNorm256(self.conv5(x)))
        x = self.activation(self.batchNorm512(self.conv6(x)))
        x = self.maxpool(x)
        
        # conv Layer-4
        x = self.activation(self.batchNorm256(self.conv7(x)))
        x = self.activation(self.batchNorm512(self.conv8(x)))
        x = self.activation(self.batchNorm256(self.conv9(x)))
        x = self.activation(self.batchNorm512(self.conv10(x)))
        x = self.activation(self.batchNorm256(self.conv11(x)))
        x = self.activation(self.batchNorm512(self.conv12(x)))
        x = self.activation(self.batchNorm256(self.conv13(x)))
        x = self.activation(self.batchNorm512(self.conv14(x)))
        x = self.activation(self.batchNorm512(self.conv15(x)))
        x = self.activation(self.batchNorm1024(self.conv16(x)))
        x = self.maxpool(x)
        
        # conv Layer-5
        x = self.activation(self.batchNorm512(self.conv17(x)))
        x = self.activation(self.batchNorm1024(self.conv18(x)))
        x = self.activation(self.batchNorm512(self.conv19(x)))
        x = self.activation(self.batchNorm1024(self.conv20(x)))
        x = self.activation(self.batchNorm1024(self.conv21(x)))
        x = self.activation(self.batchNorm1024(self.conv22(x)))
        
        # conv Layer-6
        x = self.activation(self.batchNorm1024(self.conv23(x)))
        x = self.activation(self.batchNorm1024(self.conv24(x)))
        
        # conn Layer-1
        x = x.view(x.size(0), -1)  # Flatten 
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        
        # conn Layer-2
        x = self.fc2(x)
        x = x.view(x.size(0), 7, 7, 30)
                
        return x