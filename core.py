import csv
import logging
import time

import torch
from scipy import interpolate
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import os
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from logging_file import logger

def thresholding(img: np.array):
    # 返回大于某个值的坐标
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
            self.type = 0
        elif len(target_ls) == 1 and '0' in target_ls:
            self.type = 1
        else:
            self.type = "unknowing"
            print('unknowing type ')

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
                r = Result(result, self)
                l2.append(r)
        return l2



class Result:
    def __init__(self, result,model):
        """
        一张图片的结果
        :param result:
        :param model:
        :return:None
        """
        self.model = model
        self.names = result.names
        self.cell_list = [] # 惰性
        if len(result) != 0:
            # self.pic = results[0].orig_img
            # self.orig_shape = results[0].orig_shape
            pass
        else:
            pass
        self.pic = result.orig_img
        self.orig_shape = result.orig_shape
        self.boxes = result.boxes
        self.masks = result.masks
        self.confident = result.boxes.conf


    @staticmethod
    def resize_image(image, new_shape):
        x, y = np.arange(image.shape[1]), np.arange(image.shape[0])
        f = interpolate.interp2d(x, y, image, kind='linear')
        new_x, new_y = np.linspace(0, image.shape[1], new_shape[1]), \
            np.linspace(0, image.shape[0], new_shape[0])
        resized_image = f(new_x, new_y)
        return resized_image

    def return_segmented_info(self):
        '''
        返回掩膜分割后的图像，需注意掩膜形状和原图像不完全相同
        :param tensor(count, mask_shape[0], mask_shape[1]) tensor of result
        :return: tensor(count, orig_pic_shape[0], orig_pic_shape[1]) the part of image
        '''
        count, mask_height, mask_width = self.masks.shape
        orig_height, orig_width = self.orig_shape

        # 初始化存储分割结果的张量，确保是 uint8 类型
        segmented_images = torch.zeros((count, orig_height, orig_width,3), dtype=torch.uint8)
        pic_tensor = torch.tensor(self.pic)
        results = list()
        pos_ls = list()
        for i in range(count):
            # 获取每个掩膜
            mask = self.masks.data[i]

            # 获取当前边界框，并将规范化的坐标转换为原图尺寸的实际坐标
            # 获取当前边界框的规范化坐标，确保是 float32 类型的张量
            xyxyn = self.boxes.xyxyn[i]  # 假设这是一个 tensor([x1_norm, y1_norm, x2_norm, y2_norm])

            # 获取原图的尺寸
            orig_height, orig_width = self.orig_shape  # (height, width)

            # 将规范化的坐标转换为原图尺寸的像素坐标
            x1_norm, y1_norm, x2_norm, y2_norm = xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3]

            # 转换为像素坐标 (注意: 乘以宽度/高度，确保类型为 int)
            x_min = int(x1_norm * orig_width)
            y_min = int(y1_norm * orig_height)
            x_max = int(x2_norm * orig_width)
            y_max = int(y2_norm * orig_height)

            # 裁剪原图像区域
            cropped_image = pic_tensor[y_min:y_max, x_min:x_max]  # shape: (crop_height, crop_width)

            # 将掩膜调整为原图的尺寸
            mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(orig_height, orig_width),
                                         mode='bilinear', align_corners=False)
            mask_resized = torch.round(mask_resized) #二值化
            result = mask_resized.numpy().astype(np.uint8).squeeze(0).squeeze(0)
            mask_resized = mask_resized.squeeze(0).expand(3, -1, -1)  # shape to: (channel,orig_height, orig_width)
            mask_resized = mask_resized.permute(1, 2, 0) #改顺序
            mask_resized = mask_resized[y_min:y_max, x_min:x_max]
            parted_image = cropped_image * mask_resized  # 用掩膜提取感兴趣区域
            results.append(result[y_min:y_max, x_min:x_max])
            # 将分割后的图像放回到最终结果张量中
            segmented_images[i, y_min:y_max, x_min:x_max,:] = parted_image
            pos_ls.append((x_min, y_min, x_max, y_max))
        return results,segmented_images,pos_ls

    def return_segmented_images(self):
        '''
        返回掩膜分割后的图像，需注意掩膜形状和原图像不完全相同
        :param tensor(count, mask_shape[0], mask_shape[1]) tensor of result
        :return: tensor(count, orig_pic_shape[0], orig_pic_shape[1]) the part of image
        '''
        count, mask_height, mask_width = self.masks.shape
        orig_height, orig_width = self.orig_shape

        # 初始化存储分割结果的张量，确保是 uint8 类型
        segmented_images = torch.zeros((count, orig_height, orig_width,3), dtype=torch.uint8)
        pic_tensor = torch.tensor(self.pic)
        results = list()
        pos_ls = list()
        for i in range(count):
            # 获取每个掩膜
            mask = self.masks.data[i]

            # 获取当前边界框，并将规范化的坐标转换为原图尺寸的实际坐标
            # 获取当前边界框的规范化坐标，确保是 float32 类型的张量
            xyxyn = self.boxes.xyxyn[i]  # 假设这是一个 tensor([x1_norm, y1_norm, x2_norm, y2_norm])

            # 获取原图的尺寸
            orig_height, orig_width = self.orig_shape  # (height, width)

            # 将规范化的坐标转换为原图尺寸的像素坐标
            x1_norm, y1_norm, x2_norm, y2_norm = xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3]

            # 转换为像素坐标 (注意: 乘以宽度/高度，确保类型为 int)
            x_min = int(x1_norm * orig_width)
            y_min = int(y1_norm * orig_height)
            x_max = int(x2_norm * orig_width)
            y_max = int(y2_norm * orig_height)

            # 裁剪原图像区域
            cropped_image = pic_tensor[y_min:y_max, x_min:x_max]  # shape: (crop_height, crop_width)

            # 将掩膜调整为原图的尺寸
            mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(orig_height, orig_width),
                                         mode='bilinear', align_corners=False)
            mask_resized = torch.round(mask_resized) #二值化
            result = mask_resized.numpy().astype(np.uint8).squeeze(0).squeeze(0)
            mask_resized = mask_resized.squeeze(0).expand(3, -1, -1)  # shape to: (channel,orig_height, orig_width)
            mask_resized = mask_resized.permute(1, 2, 0) #改顺序
            mask_resized = mask_resized[y_min:y_max, x_min:x_max]
            parted_image = cropped_image * mask_resized  # 用掩膜提取感兴趣区域
            results.append(result[y_min:y_max, x_min:x_max])
            # 将分割后的图像放回到最终结果张量中
            segmented_images[i, y_min:y_max, x_min:x_max,:] = parted_image
            pos_ls.append((x_min, y_min, x_max, y_max))
        return results,segmented_images,pos_ls

    def return_cell(self):
        """
        返回图像中的细胞对象的列表
        :return: list[object:cell]
        """
        if len(self.cell_list) == 0:
            if self.model.type == 0:
                # self.connect_masks()
                pass
            elif self.model.type == 1:
                masks,seg_img,pos_ls = self.return_segmented_images()
                for i in range(seg_img.shape[0]):
                    c = cell(masks[i],
                             self.pic,
                             pos_ls[i],
                             conf = self.confident[i])
                    self.cell_list.append(c)
        return self.cell_list


class cell:
    def __init__(self,*seg, **params):
        '''
        seg,orig,mask
        :param seg:tuple(np.array)
        :param params:
        '''
        self.seg = seg[0]
        self.orig = seg[1]
        self.pos = seg[2]
        self.params = params

        self.contour = None
        self.stomata_PCA = None
        self.cell_PCA = None
        self.rate = None
        self.stomata_area = None
        self.cell_area = None
        self.is_open = None
        self.cell = None
        self.stomata = None


    @staticmethod
    def return_PCA_result(mask: np.array) -> object:
        """
        返回PCA分析结果,输入为mask
        :return:class PCA_result
        """
        # 性能瓶颈
        start_time = time.time()
        X = np.array(np.where(mask > 0.2))
        if X.shape[1] == 0:
            raise ValueError("No valid coordinates found in the mask array (mask > 0.2).")
        X = np.transpose(X)
        mean_X = np.mean(X[:, 1])
        mean_Y = np.mean(X[:, 0])
        pca = PCA(n_components=1)
        if X.shape[0] < 2:
            raise ValueError("Insufficient data points for PCA (need at least 2 samples).")
        X_pca = pca.fit_transform(X)
        main = pca.components_  # 主成分方向向量
        end_time = time.time()
        logger.debug(f"PCA的运行时间: {end_time - start_time} 秒")

        axis = X_pca  # 降维后数据
        line = np.dot(axis, main)
        line_x = line[:, 1] + mean_X
        line_y = line[:, 0] + mean_Y
        return PCA_result(axis, main, line_x, line_y)


    def analyse_type2(self,bit_map):
        """
        第二种分析方式：
        识别出的一个气孔
        如果错误，PCA会为None
        type1
        .rate:面积占比
        :param bit_map: orig形状的mask
        :param seg:保卫细胞分割图（掩膜）
        :param seg2:气孔分割图（掩膜）
        :param is_open:
        """


        self.cell = self.seg
        start_time3 = time.time()
        try:
            # if self.is_open:
            cell_result = self.return_PCA_result(bit_map)
        except TypeError as e:
            cell_result = None
            logger.debug("return_PCA_result失败,{}".format(e))
            print(e)
        end_time4 = time.time()
        self.cell_PCA = cell_result
        cell_area = np.sum(bit_map)  # 求和，计算非零像素点的总数
        self.cell_area = cell_area
        temp_255 = bit_map * 255
        contours, hierarchy = cv.findContours(temp_255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        perimeter = -1
        for contour in contours:
            perimeter = cv.arcLength(contour, True)  # 计算轮廓的周长
            break
        self.contour = perimeter

        logger.debug(f"分析中PCA_result的运行时间: {end_time4 - start_time3} 秒")




    def return_bit(self):
        # 返回二值图
        result = np.zeros(self.orig.shape[0:2], dtype=np.uint8)
        x_min, y_min, x_max, y_max = self.pos
        result[y_min:y_max, x_min:x_max] = self.seg
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

