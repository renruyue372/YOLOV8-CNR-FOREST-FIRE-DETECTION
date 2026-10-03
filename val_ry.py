from ultralytics import YOLO


if __name__ == '__main__':
    #model = YOLO("D:\\fjx\\ultralyticsPro0401-YOLOv8\\external_baselines\\fireyolo_lite\\outputs\\fireyolo_lite_300e_b16_w8\\train\\train_300e_b16_w8\\weights\\best.pt")
    model = YOLO("D:\\fjx\\ultralyticsPro0401-YOLOv8\\rynew\\train\\weights\\best.pt")
    print(model.info())
    model.val(batch=16,workers=0,device=0
    )
