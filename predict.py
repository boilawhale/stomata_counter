import csv

from scipy import interpolate
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import os
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 避免中文乱码
matplotlib.rc("font", family='YouYuan')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def thresholding(img: np.array):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    X = np.array(np.where(img > 0.2))
    X = np.transpose(X)
    return X

'''
模型1：只定位气孔
model.model.task = 'detect'
模型2：气孔分割，保卫细胞分割，是否开闭
model.names = dict(3)
model.model.task = 'segment'
模型3：气孔分割
model.names = dict(1)
'''
class Model:
    def __init__(self, model_path='runs/detect/train5/weights/best.pt', root_path='./'):
        self.root_path = root_path
        self.model = YOLO(root_path + model_path)
        self.name = self.model.names
        self.pic = None
        target_ls = self.model.names.values()  #dict
        if len(target_ls) == 3 and 'cell' in target_ls and 'close' in target_ls and 'open' in target_ls:
            self.type = "1"
        else:
            self.type = "unknow"

    def _get_data(self, path):
        """
        :param path:
        :return:list[image object]
        """
        if not os.path.exists(path):
            path = self.root_path + path
        if os.path.isfile(path):
            if path.endswith(('.bmp', '.jpg', '.jpeg')):
                img = Image.open(path)
                return [img]
            else:
                raise TypeError('错误的图像格式')
        elif os.path.isdir(path):
            d = os.listdir(path)
            ls = list()
            for i in d:
                p = os.path.join(path, i)
                if os.path.isfile(p) and i.endswith(('.bmp', '.jpg', '.jpeg')):
                    ls.append(p)
            return ls
        else:
            raise TypeError('错误的路径')

    def train(self, dataset_name='origin', epochs=1000, device=0, batch=32):
        self.model.train(data='./dataset/{}/data.yaml'.format(dataset_name), epochs=epochs, device=device, batch=batch)

    def return_results(self, path='20230627_184242_2.bmp'):
        '''
        返回模型的预测结果
        :param path:从根目录下得文件路径或单个文件
        :return:Result object list理论可多张图片
        '''
        l = self._get_data(path)
        l2 = list()
        for f in l:
            results = self.model.predict(source=f, save=False)
            for result in results:
                result = result.cpu()
                result = result.numpy()
                r = Result(result, self)
                l2.append(r)
        return l2

    @staticmethod
    def resize_image(image, new_shape):
        x, y = np.arange(image.shape[1]), np.arange(image.shape[0])
        f = interpolate.interp2d(x, y, image, kind='linear')
        new_x, new_y = np.linspace(0, image.shape[1], new_shape[1]), \
            np.linspace(0, image.shape[0], new_shape[0])
        resized_image = f(new_x, new_y)
        return resized_image

    def save_segmented_images(self, output_dir: str, save_path='pic/stomata/'):
        """
        保存分割出的图像为彩色图。

        Args:
            results (Results): YOLOv8 分割模型生成的 results 类。
            orig_image (np.ndarray): 原始彩色图像。
            output_dir (str): 保存图像的目标文件夹路径。
            :param output_dir:
            :param save_path:
        """
        pat = self.return_results('dataset/pic/20230627_194422_12_bmp.rf.cf9054da86fc2cbf25f2ada017d8c2ca.jpg')
        save_path = self.root_path + save_path

        for results in pat:
            pic_name = results.path
            pic_name = os.path.splitext(pic_name)[1]
            orig_image = deepcopy(results.orig_img)
            masks1 = results.masks
            masks2 = masks1.cpu()
            masks3 = masks2.data
            masks4 = masks3.numpy()
            for i, mask in enumerate(masks4):
                mask = self.resize_image(mask, (768, 1024))
                # 提取掩码对应的部分
                o = np.expand_dims(mask, axis=-1)
                segmented_part = orig_image * o
                segmented_part = segmented_part.astype(np.uint8)

                # 保存为新的图像文件
                filename = f"segmented_{i}.png"
                output_path = os.path.join('pic/segment/', filename)
                cv.imwrite(output_path, segmented_part)

    def show_edges(self, path='dataset/pic/20230627_194422_12_bmp.rf.cf9054da86fc2cbf25f2ada017d8c2ca.jpg'):
        '''
        展示边缘检测后的图片
        :param path:单个文件位置
        :return: None
        '''
        a = self.return_results(path)
        if a[0] != None:
            for i in a[0]:
                s = i.boxes.xyxy
                s = s.cpu().numpy()
                pic = i.orig_img
                for j in s[:]:
                    j1, j2, j3, j4 = int(j[1]), int(j[3]), int(j[0]), int(j[2])
                    r = pic[j1:j2, j3:j4]
                    k = cv.cvtColor(r, cv.COLOR_BGR2GRAY)
                    edges = cv.Canny(k, 100, 200)
                    cv.imshow('2', edges)
                    cv.waitKey(5000)
                    cv.destroyAllWindows()

    def show_results_windows(self):
        """
        以窗口展示结果
        :return:
        """

    def save_csv_file(self):
        """
        保存csv文件
        :return:
        """


class Result:
    def __init__(self, results,model):
        """
        一张图片的结果
        :param results:
        """
        self.model = model
        self.names = results.names
        self.cell_list = []
        if len(results) != 0:
            self.pic = results[0].orig_img
            self.orig_shape = results[0].orig_shape
        else:
            pass
        self.box = []
        self.masks = []
        for result in results:
            self.box.append(result.boxes)
            self.masks.append(result.masks)

    def connect_masks(self):
        '''
        根据相关关系判断是否采用气孔识别的结果
        :return:
        '''
        for i, box in enumerate(self.box):
            if box.cls[0] == 0:
                for j, box2 in enumerate(self.box):
                    if self.is_inclusive(box, box2):
                        box_ = self.return_segmented_images(i)
                        box2_ = self.return_segmented_images(j)
                        op = 0
                        if box2.cls[0] == 2:
                            op = 1
                        c = cell(box_, box2_, op)
                        self.cell_list.append(c)
                        break  # 可做个置信度比较

    @staticmethod
    def is_inclusive(box1, box2) -> bool:
        a = 1
        if box1.xyxy[0, 0] <= box2.xyxy[0, 0]*a and \
                box1.xyxy[0, 1] <= box2.xyxy[0, 1]*a and \
                box1.xyxy[0, 2] >= box2.xyxy[0, 2]*a and \
                box1.xyxy[0, 3] >= box2.xyxy[0, 3]*a and \
                box1.cls[0] == 0 and \
                box2.cls[0] != 0:
            return True
        else:
            return False

    @staticmethod
    def resize_image(image, new_shape):
        x, y = np.arange(image.shape[1]), np.arange(image.shape[0])
        f = interpolate.interp2d(x, y, image, kind='linear')
        new_x, new_y = np.linspace(0, image.shape[1], new_shape[1]), \
            np.linspace(0, image.shape[0], new_shape[0])
        resized_image = f(new_x, new_y)
        return resized_image

    def return_segmented_images(self, index):
        for i, mask in enumerate(self.masks[index]):
            mask2 = np.array(mask.data)[0]
            mask3 = self.resize_image(mask2, self.orig_shape)
            # 提取掩码对应的部分
            o = np.expand_dims(mask3, axis=-1)
            segmented_part = self.pic * o
            segmented_part = segmented_part.astype(np.uint8)
        return segmented_part

    def cut_cell(self):
        pass

    def show_result(self):
        self.connect_masks()
        img = self.pic
        c = None
        clo = None
        ope = None
        mean = []
        t = 1
        for cell_ in self.cell_list:
            cell_.analyse()
            plt.subplot(2, 2, 1)
            plt.scatter(cell_.cell_PCA.line_x, cell_.cell_PCA.line_y, c="r")
            if cell_.is_open == 1:
                plt.scatter(cell_.stomata_PCA.line_x, cell_.stomata_PCA.line_y, c="g")
            else:
                plt.scatter(cell_.stomata_PCA.line_x, cell_.stomata_PCA.line_y, c="b")

            if c is None:
                c = cv.cvtColor(cell_.cell, cv.COLOR_BGR2GRAY)
            else:
                c += cv.cvtColor(cell_.cell, cv.COLOR_BGR2GRAY)

            if cell_.is_open == 1:
                if clo is None:
                    clo = cv.cvtColor(cell_.stomata, cv.COLOR_BGR2GRAY)
                else:
                    clo += cv.cvtColor(cell_.stomata, cv.COLOR_BGR2GRAY)

            else:
                if ope is None:
                    ope = cv.cvtColor(cell_.stomata, cv.COLOR_BGR2GRAY)
                else:
                    ope += cv.cvtColor(cell_.stomata, cv.COLOR_BGR2GRAY)
            mean.append(cell_.rate)
            if t <= 4:
                plt.subplot(4, 4, 4 * t)
                sns.kdeplot(cell_.cell_PCA.axis[:, 0], c='r')
                sns.kdeplot(cell_.stomata_PCA.axis[:, 0], c='g')
                plt.title(f"气孔平均占比为{cell_.rate}")
                t += 1

        if len(mean) != 0:
            mean_rate = sum(mean) / len(mean)

        if img is not None:
            c = np.where(c > 0, 1, 0)
            img[c == 1] = [200, 127, 127]

        if clo is not None:
            clo = np.where(clo > 0, 1, 0)
            img[clo == 1] = [127, 127, 127]
        if ope is not None:
            ope = np.where(ope > 0, 1, 0)
            img[ope == 1] = [127, 127, 200]
        # c = c.astype(np.uint8)
        # s = s.astype(np.uint8)
        # c = c * 50
        # s = s * 50
        #
        # img[:, :, 0] += c
        # img[:, :, 1] += s

        # img[:, :, 0][img[:, :, 0] > 250] = 250
        # img[:, :, 1][img[:, :, 1] > 250] = 1
        # img[:, :, 2][img[:, :, 2] > 250] = 250
        plt.suptitle(f"气孔占比为{mean_rate}")
        plt.imshow(img)

        plt.show()

    def return_cell(self):
        """
        返回图像中的细胞对象的列表
        :return: list[object:cell]
        """
        if len(self.cell_list) == 0:
            # if self.model.names
            self.connect_masks()
        return self.cell_list


class cell:
    def __init__(self, seg: np.array, seg2: np.array, is_open):
        """
        识别出的一个气孔
        .rate:面积占比
        :param seg:保卫细胞分割图（掩膜）
        :param seg2:气孔分割图（掩膜）
        :param is_open:
        """
        self.stomata_PCA = None
        self.cell_PCA = None
        self.rate = None
        self.stomata_area = None
        self.cell_area = None
        self.is_open = is_open
        self.cell = seg
        self.stomata = seg2
        self.analyse()

    @staticmethod
    def return_PCA_result(k: np.array) -> object:
        """
        返回PCA分析结果
        :return:class PCA_result
        """
        X = thresholding(k)
        mean_X = np.mean(X[:, 1])
        mean_Y = np.mean(X[:, 0])
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X)
        main = pca.components_  # 主成分方向向量
        axis = X_pca  # 降维后数据
        line = np.dot(axis, main)
        line_x = line[:, 1] + mean_X
        line_y = line[:, 0] + mean_Y
        return PCA_result(axis, main, line_x, line_y)

    def analyse(self):
        cell_result = self.return_PCA_result(self.cell)
        stomata_result = self.return_PCA_result(self.stomata)
        cell_area = np.sum(thresholding(self.cell))
        stomata_area = np.sum(thresholding(self.stomata))
        rate = stomata_area / (cell_area + stomata_area)
        self.rate = rate
        self.cell_PCA = cell_result
        self.stomata_PCA = stomata_result
        self.cell_area = cell_area
        self.stomata_area = stomata_area

    def return_bit(self):
        stomata = cv.cvtColor(self.stomata, cv.COLOR_BGR2GRAY)
        cell_ = cv.cvtColor(self.cell, cv.COLOR_BGR2GRAY)
        bit = cell_+stomata
        result = np.where(bit > 0, 1, 0)
        return result


class PCA_result:
    def __init__(self, axis, main, line_x, line_y):
        """
        单个气孔的主成分分析结果
        .axis 降维后数据，ndarray(n,1)
        .line_x 降维后点的x坐标
        .line_y 降维后点的y坐标
        .main 主成分方向向量，ndarray(1,2)
        """
        self.axis = axis
        self.main = main
        self.line_x = line_x
        self.line_y = line_y

        self.max_axis_length = max(axis) - min(axis)
        plt.subplot(4, 4, 13)
        nums, p, o = plt.hist(axis)
        self.max_wide = max(nums)
        plt.cla()

    def return_axis(self):
        """
        返回降维后数据
        :return: list
        """
        return self.axis

    def return_main(self):
        """
        返回主成分方向向量
        :return: list
        """

    def return_max_axis_length(self):
        """
        返回长轴长度
        :return:float
        """
        return self.max_axis_length

    def return_max_wide(self):
        """
        返回图像最大宽度
        :return:float
        """
        return self.max_wide


def type1():
    a = Model(model_path='disuse/segment/train17/weights/best.pt')
    # 设置保存目录
    output_directory = 'pic/segment/'

    # 调用函数保存分割图像
    a.save_segmented_images(output_dir=output_directory)


# 展示一个图片
def type2():
    a = Model(model_path='disuse/segment/train17/weights/best.pt')
    c = a.return_results(path='20230627_201522_29_bmp_jpg.rf.1b8c85f064148ffe5e9147bbe0cdc494.jpg')
    i = c[0]
    i = i.cpu()
    i = i.numpy()
    a = Result(i)
    a.show_result()


# 展示许多图片
def type3():
    mod = Model(model_path='runs/segment/train10 extend/weights/best.pt')
    c = mod.return_results(path='20230627_201522_29_bmp_jpg.rf.1b8c85f064148ffe5e9147bbe0cdc494.jpg')
    for i in c:
        i.show_result()


def predict_as_csv(dir_path):
    name = os.path.split(dir_path)[1]
    with open(name + 'result.csv', "w", encoding='utf8', newline='') as f:
        a = csv.writer(f, )
        a.writerow(('气孔面积', '保卫细胞面积', '是否打开', '保卫细胞方向x', ' 保卫细胞方向y', '气孔方向x',
                    '气孔方向y', '面积占比', '属于的叶片（文件路径）', '叶子序数', '气孔序数'))
    mod = Model(model_path='runs/segment/train10 extend/weights/best.pt')
    n_of_leaf = 0
    for i0 in os.listdir(dir_path):
        n = os.path.join(dir_path, i0)
        if os.path.isdir(n):
            for i in os.listdir(n):
                if i.endswith('.bmp'):
                    ls = return_PCA_result_list(n, i, mod)
                    with open(name + 'result.csv', "a", encoding='utf8', newline='') as f:
                        a = csv.writer(f, )
                        n_of_pic = 0
                        for r in ls:
                            r = list(r)
                            r.append(n_of_leaf)
                            r.append(n_of_pic)
                            a.writerow(r)
                            n_of_pic += 1
                    n_of_leaf += 1


def return_PCA_result_list(dir_path, filename, mod):
    '''
    单张图片的
    :param dir_path:file location
    :param filename:str
    :param mod: mode
    :return:list
    '''
    filepath = os.path.join(dir_path, filename)
    c = mod.return_results(filepath)
    i = c[0]
    i.connect_masks()
    ls = []
    for j in i.cell_list:
        # 气孔面积  保卫细胞面积  是否打开 保卫细胞方向x 保卫细胞方向y    气孔方向x 气孔方向y 面积占比  属于的叶片（文件路径）
        data_set = (
            j.stomata_area,
            j.cell_area,
            j.is_open,
            j.cell_PCA.main[0, 0],
            j.cell_PCA.main[0, 1],
            j.stomata_PCA.main[0, 0],
            j.stomata_PCA.main[0, 1],
            j.rate,
            filepath
        )
        ls.append(data_set)
    return ls


if __name__ == '__main__':
    # a = Model(model_path='disuse/segment/train17/weights/best.pt')
    # a.model.val()
    # type3()
    type3()
