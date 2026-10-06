import sys
import argparse
import os

from paddle.optimizer import AdamW
from timm.data import auto_augment_policy
from torch.nn.functional import dropout

sys.path.append(r'E:\GitHubRepo\PR\ultralyticsPro-') # Path

from ultralytics import YOLO

'''
python train_v8.py --cfg 

'''

def main(opt):
    yaml = opt.cfg
    #weights = opt.weights
    # model = YOLO(weights)
    model = YOLO(yaml)


    # print(model)
    # 87.3% 87.8%(+A) 87.5%(+A+B)

    model.info()

    results = model.train(data='D:\\fjx\\ultralyticsPro0401-YOLOv8\\M4SFWD Dataset New\\data.yaml',
                          epochs=300,
                          batch = 32,
                          imgsz = 640,
                          project='rynew',  # 自定义项目路径
                          name='train',  # 自定义实验名称
                          optimizer = 'SGD',
                          lr0 = 0.01,
                          lrf = 0.1,
                          momentum = 0.937,
                          weight_decay = 0.0003,
                          box = 8.0,
                          cls = 0.8,
                          dfl = 5.0,
                          cos_lr = True,
                          warmup_epochs = 12,
                          hsv_h = 0.05,
                          hsv_s = 0.7,
                          hsv_v = 0.4,
                          mosaic = 0.3,
                          mixup = 0.05,
                          copy_paste = 0.1,
                          pretrained = True,
                          seed = 123
                          )



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\v10\yolov8n.yaml', help='initial weights path')


    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)