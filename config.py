from pathlib import Path


root = Path("./VOCdata")
year = "2012"
image_set_train = "train"  # 사용할 데이터셋 유형 (train, trainval, val)
image_set_trainval = "trainval"
image_set_val = "val"
download = True  # 데이터셋 다운로드 여부

# S x S x (B*5 + C) = 7 x 7 x 30 tensor
grid_S = 7
box_B = 2
class_C = 20

lambda_coord = 5
lambda_noobj = 0.5

total_epoch = 135
batch_size = 24
momentum = 0.9
weight_decay = 0.0005
learning_rate_init = 0.001   # up to 0.01 , epoch 75
learning_rate_2 = 0.001      # epoch 76~105
learning_rate_3 = 0.0001     # epoch 106~135
drop_out_rate = 0.5
data_augmentation = True
scale_translation = 0.2
exposure_saturation = 1.5



