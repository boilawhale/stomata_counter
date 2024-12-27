import tkinter as tk
from tkinter import filedialog

import numpy as np

import core
from PIL import Image, ImageTk
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 避免中文乱码
matplotlib.rc("font", family='YouYuan')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# model_path = 'runs/segment/train10 extend/weights/best.pt'
model_path = 'runs/segment/train34/weights/best.pt'


ls_of_photo = []
ls_of_time = []
dir_of_result = dict()
dir_of_pic_cache = dict()
index = 0
default_picdir = r'C:\Users\Mr zhang\Desktop\张霄\2023.0817\10X反'

first_open = True
Model = core.Model(model_path=model_path, root_path='./')

class control_ls():
    def __init__(self,path):
        self.dir_path = path
        self.ls_of_time = []
        self.deal_pics_cache = dict()

    def add_new_pic(self,path):
        results = Model.return_results(path)
        result = results[0]
        new_pic = dealt_pic_cache(result)
        self.ls_of_time.append(path)
        self.deal_pics_cache[path] = new_pic

    def remove_pic(self):
        pass

    def garbage_collection(self):
        pass

    def save_all_data_as_csv(self):
        pass



class dealt_pic_cache():
    '''
    处理过的类
    '''
    def __init__(self,orig:core.Result):
        self.source = orig
        self.mask_ls = list()
        orig_pic = orig.pic
        cell_list = orig.return_cell()
        mask_layer = np.zeros(orig.orig_shape)
        a = 0.5  # 透明度
        color_mask = np.zeros([orig.orig_shape[0],
                               orig.orig_shape[1],
                               3
                               ],
                              dtype=float
                              )
        # mask的颜色
        color_mask[:, :, 0] = 240
        color_mask[:, :, 1] = 140
        color_mask[:, :, 2] = 50

        for i in cell_list:
            mask_layer += i.return_bit()
            self.mask_ls.append(mask_layer)
        mask_layer[mask_layer > 1] = 1
        mask_layer = np.expand_dims(mask_layer, axis=2)
        mask_layer = np.repeat(mask_layer, 3, axis=2)
        color_mask = color_mask * mask_layer
        mask_layer[mask_layer == 1] = a
        mask_layer[mask_layer == 0] = 1

        res = mask_layer * orig_pic + color_mask * (1 - a)
        res = res.astype('uint8')
        self.orig_pic = orig_pic
        self.masked_pic = res

    def update_result(self,orig):
        mask_layer = np.zeros(orig.orig_shape)
        for i in self.mask_ls:
            mask_layer += i
        self.masked_pic = mask_layer

    def add_mask(self,mask:np.array):
        pass
    def remove_mask(self,id):
        pass
class Mask():
    pass


class showing_pic():
    def __init__(self,dealt_pic:dealt_pic_cache):
        self.showing_img = dealt_pic
        self.show_mask = list()
        self.dealt_pic = dealt_pic
        self.width = 0
        self.height = 0

    def change_to(self,img:np.array,dealt_pic):
        self.showing_img = img
        self.dealt_pic = dealt_pic
        self.show()

    def return_showing_mask_id_by_pos(self,pos:tuple[int]):
        pass

    def change_remove_one_result_by_id(self,i):
        self.show_mask.remove(i)
        self.show()

    def add_poly(selfself):
        pass

    def show(self):
        pass



class data_analysis():
    pass


def show(path):
    if is_mask == 0:
        img = img_deal(path)
        main_img.config(image=img)
    else:
        if path in dir_of_result:
            pass
        else:
            result = Model.return_results(path)
            ls_of_time.append(path)
            dir_of_result[path] = result[0]

        result_of_pic = dir_of_result[path]
        orig_pic = result_of_pic.pic
        cell_list = result_of_pic.return_cell()
        mask = np.zeros(result_of_pic.orig_shape)
        a = 0.5  # 透明度
        color_mask = np.zeros([result_of_pic.orig_shape[0],
                               result_of_pic.orig_shape[1],
                               3
                               ],
                              dtype=float
                              )
        # mask的颜色
        color_mask[:, :, 0] = 240
        color_mask[:, :, 1] = 140
        color_mask[:, :, 2] = 50
        if True:
            for i in cell_list:
                mask += i.return_bit()
            mask[mask > 1] = 1
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            color_mask = color_mask * mask
            mask[mask == 1] = a
            mask[mask == 0] = 1

        res = mask * orig_pic + color_mask * (1 - a)
        res = res.astype('uint8')
        img = img_deal(res)

        main_img.config(image=img)


def next():
    global index
    if len(ls_of_photo) == 0:
        return None
    elif index + 1 == len(ls_of_photo):
        index = 0
    elif index < len(ls_of_photo):
        index += 1
    show(ls_of_photo[index])
    change_show_index()


def last():
    global index
    if len(ls_of_photo) == 0:
        return None
    elif index == 0:
        index = len(ls_of_photo) - 1
    elif index > 0:
        index -= 1
    show(ls_of_photo[index])
    change_show_index()


is_mask = 1


def mask():
    global is_mask
    if is_mask == 0:
        is_mask = 1
        show(ls_of_photo[index])
    elif is_mask == 1:
        is_mask = 0
        show(ls_of_photo[index])


def open_dir(origin=default_picdir):
    global ls_of_photo
    global index
    global first_open
    # 选择文件夹
    if first_open:
        folder_path = origin
        first_open = False
    else:
        folder_path = tk.filedialog.askdirectory()
    if folder_path:
        # 清空列表
        ls_of_photo.clear()
        # 遍历文件夹，收集文件名
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.bmp')):
                ls_of_photo.append(os.path.join(folder_path, filename))
    if len(ls_of_photo) > 0:
        show(ls_of_photo[0])
        index = 0
        change_show_index()
        for i in ls_of_photo:
            result = Model.return_results(i)
            ls_of_time.append(i)
            dir_of_result[i] = result[0]
    else:
        pass


def change_show_index():
    total = len(ls_of_photo)
    t = index + 1
    label_of_index.config(text="图片数量\n{}--{}".format(t, total))


def img_deal(input):
    if isinstance(input, str):
        img = Image.open(input)
    elif isinstance(input, np.ndarray):
        img = input.astype(np.uint8)
        img = Image.fromarray(img)
    else:
        print(type(input))
        pass
    global img2_avoid_trash  # 防止返回就被垃圾回收了
    img2_avoid_trash = ImageTk.PhotoImage(img)
    return img2_avoid_trash

def left_chick_of_img(event):
    pass


windows_width = 1350
windows_height = 768

root = tk.Tk()
root.geometry(str(windows_width) + "x" + str(windows_height) + "+0+0")
root.title("111")
main_img = tk.Label(root,
                    bg='black',
                    height=768, width=1024,
                    padx=0, pady=0
                    )
f1 = tk.Frame(root,
              width=100, height=768,
              padx=10, pady=10, borderwidth=5,
              bd=8, bg='white',
              relief='groove')
f1.pack(side='left')
main_img.pack()
next_button = tk.Button(f1,
                        width=6, height=2,
                        relief='raise',
                        command=next,
                        text='下一张'
                        )
next_button.pack()
last_button = tk.Button(f1,
                        width=6, height=2,
                        relief='raise',
                        command=last,
                        text='上一张')
last_button.pack()
open_dir_button = tk.Button(f1,
                            width=6, height=2,
                            relief='raise',
                            command=open_dir,
                            text='打开文件夹')
open_dir_button.pack()
show_button = tk.Button(f1,
                        width=6, height=2,
                        relief='raise',
                        command=mask,
                        text='显示隐藏')
show_button.pack()
label_of_index = tk.Label(f1,
                          width=6, height=2,
                          relief='flat',
                          command=None,
                          text='')
label_of_index.pack()


def update_text():
    # 更新文本内容
    # 每1000毫秒（1秒）调用一次update_text函数
    root.after(1000, update_text)
open_dir()
root.mainloop()

