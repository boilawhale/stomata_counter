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

model_path = 'runs/segment/train10 extend/weights/best.pt'
model_path2 = 'runs/segment/train16/weights/best.pt'


ls_of_photo = []
ls_of_time = []
dir_of_result = dict()
dir_of_pic_cache = dict()
index = 0
default_picdir = r'C:\Users\Mr zhang\Desktop\张霄\2023.0817\10X反'

first_open = True
Model = core.Model(model_path=model_path2, root_path='./')

class pic_cache():
    def __init__(self, path):
        self.path = path
        self.orig = None
        self.masked = None
        self.result = None
    def set_orig(self,orig):
        self.orig = orig

    def set_mask(self,masked):
        self.masked = masked
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


path = r'G:\sc\sc-master\20230627_184242_2.bmp'
width = 1350
height = 768

root = tk.Tk()
root.geometry(str(width) + "x" + str(height) + "+0+0")
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

open_dir()
root.mainloop()
