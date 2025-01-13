import os
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train():
    model = YOLO(r'G:\sc\sc-master\runs\segment\train32\weights\best.pt ')
    model.train(data=r"G:\sc\sc-master\test-4\data.yaml", epochs=1000, device=0, batch=4)

def new_train():
    model = YOLO(r'G:\sc\sc-master\yolo\yolov8n-seg.pt')
    model.train(data=r"G:\sc\sc-master\dataset\test-3\data.yaml", epochs=1000, device=0, batch=4)

def train0(set_name='version2'):
    # 加载预训练模型
    model = YOLO("runs/segment/train29/weights/best.pt")
    model.train(data='./dataset/{}/data.yaml'.format(set_name), epochs=1000, device=0, batch=4)

def train1(dataset_path=r'/mnt/data_hdd1/yangj/zx_yolo/test-4'):
    model = YOLO(r'G:\sc\sc-master\yolo\yolov8n-seg.pt')
    model.train(data=dataset_path, epochs=1000, device=0, batch=16)



if __name__ == '__main__':
    # new_train()
    train()
    # basenum = 0
    # model = YOLO(f'G:/sc/sc-master/runs/segment/train4/weights/  j
    # while True:
    #     try:
    #         model.train(data="G:/sc/sc-master/dataset/segv1/data.yaml", epochs=1000, device=0, batch=8, resume=True)
    #         break
    #     except:
    #         basenum += 1
    #         if basenum > 10:
    #             break

    # basenum = 0
    # model = YOLO('yolov8n-seg.pt')
    # while True:
    #     try:
    #         model.train(data="G:/sc/sc-master/dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    #         break
    #     except:
    #         basenum += 1
    #         if basenum > 10:
    #             break

    # basenum = 0
    # model = YOLO('yolov8s-seg.pt')
    # while True:
    #     try:
    #         model.train(data="G:/sc/sc-master/dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    #         break
    #     except:
    #         basenum += 1
    #         if basenum > 10:
    #             break
    #
    # basenum = 0
    # model = YOLO('yolov8l-seg.pt')
    # while True:
    #     try:
    #         model.train(data="G:/sc/sc-master/dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    #         break
    #     except:
    #         basenum += 1
    #         if basenum > 10:
    #             break
    #
    # basenum = 0
    # model = YOLO('yolov8m-seg.pt')
    # while True:
    #     try:
    #         model.train(data="G:/sc/sc-master/dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    #         break
    #     except:
    #         basenum += 1
    #         if basenum > 10:
    #             break5

    # model = YOLO('yolov8n-seg.pt')
    # model.train(data="dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    # model1 = YOLO('yolov8l-seg.pt')
    # model1.train(data="dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    # model2 = YOLO('yolov8m-seg.pt')
    # model2.train(data="dataset/segv1/data.yaml", epochs=1000, device=0, batch=8)
    # model = YOLO('yolov8n-seg.pt')
    # model.train(data="G:/sc/sc-master/dataset/segv_extend/data.yaml", epochs=1000, device=0, batch=8)
    # model2 = YOLO('runs/segment/train11/weights/best.pt')
    # model2.train(data="dataset/seg_from_roboflow1/seg_from_roboflow/data.yaml", epochs=1000, device=0, batch=8)
