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

index = 0
default_picdir = r'C:\Users\Mr zhang\Desktop\张霄\2023.0817\10X反'

first_open = True
Model = core.Model(model_path=model_path, root_path='./')

class control_ls():
    def __init__(self,path = None):
        self.dir_path = path
        self.ls_of_photo = []
        self.ls_of_time = []
        self.deal_pics_cache = dict()
        self.index = 0              # 当前图片索引
        self.is_show = 1            # 是否显示遮罩
        self.first_open = True      # 是否为第一次打开文件夹

    def get_name(self,folder_path):
        self.dir_path = folder_path
        # 遍历文件夹，收集文件名
        for filename in os.listdir(self.dir_path):
            if filename.lower().endswith(('.jpg', '.bmp')):
                self.ls_of_photo.append(os.path.join(self.dir_path, filename))

    def show(self):
        pass

    def change_dir(self,new_path):
        self.dir_path = new_path

    def clear(self):
        self.ls_of_time.clear()
        self.deal_pics_cache.clear()
        self.ls_of_photo.clear()

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
    if content.is_show == 0:
        img = img_deal(path)
        main_img.config(image=img)
    else:
        if path in content.deal_pics_cache:
            pass
        else:
            result = Model.return_results(path)
            content.ls_of_time.append(path)
            content.deal_pics_cache[path] = result[0]

        result_of_pic = content.deal_pics_cache[path]
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

        color_axis = np.zeros([result_of_pic.orig_shape[0],
                               result_of_pic.orig_shape[1],
                               3
                               ],
                              dtype=float
                              )
        #axis的颜色
        color = [150,50,240]
        res =orig_pic
        if show_mask:
            for i in cell_list:
                mask += i.return_bit()
            mask[mask > 1] = 1
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            color_mask = color_mask * mask
            mask[mask == 1] = a
            mask[mask == 0] = 1
            res = mask * res + color_mask * (1 - a)

        total_area = 0
        total_perimeter = 0
        for i in cell_list:
            i.analyse_type2()
            total_area += i.cell_area
            total_perimeter += i.contour
            if show_axis:
                y = np.clip(i.cell_PCA.line_y.astype(np.int32), 0, result_of_pic.orig_shape[0]-1)
                x = np.clip(i.cell_PCA.line_x.astype(np.int32), 0, result_of_pic.orig_shape[1]-1)
                res[y,x] = color

        res = res.astype('uint8')
        img = img_deal(res)

        main_img.config(image=img)

        show_text = f'''
            气孔数量:{len(cell_list)}
            气孔面积平均面积：{total_area/len(cell_list)}
            气孔平均周长：{total_perimeter/len(cell_list)}
        '''
        label_of_number.config(text = show_text)



def next():
    if len(content.ls_of_photo) == 0:
        return None
    elif content.index + 1 == len(content.ls_of_photo):
        content.index = 0
    else:
        content.index += 1
    show(content.ls_of_photo[content.index])
    change_show_index()

def last():
    if len(content.ls_of_photo) == 0:
        return None
    elif content.index == 0:
        content.index = len(content.ls_of_photo) - 1
    else:
        content.index -= 1
    show(content.ls_of_photo[content.index])
    change_show_index()

def mask():
    if content.is_show == 0:
        content.is_show = 1
        show(content.ls_of_photo[content.index])
    elif content.is_show == 1:
        content.is_show = 0
        show(content.ls_of_photo[content.index])

def open_dir(origin=default_picdir):
    # 选择文件夹
    if content.first_open:
        folder_path = origin
        content.first_open = False
    else:
        folder_path = tk.filedialog.askdirectory()
    if folder_path:
        # 清空列表
        content.clear()
        content.get_name(folder_path)

    if len(content.ls_of_photo) > 0:
        show(content.ls_of_photo[0])
        content.index = 0
        change_show_index()
        for i in content.ls_of_photo:
            result = Model.return_results(i)
            content.ls_of_time.append(i)
            content.deal_pics_cache[i] = result[0]
    else:
        pass

def change_show_index():
    total = len(content.ls_of_photo)
    t = content.index + 1
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

f2 = tk.Frame(root,
              width=400, height=768,
              padx=10, pady=10, borderwidth=5,
              bd=8, bg='white',
              relief='groove')
f2.pack(side='right')

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

show_mask = tk.BooleanVar(value=True)  # 用于存储复选框的值
check_button = tk.Checkbutton(f1, text="启用遮罩", variable=show_mask)
check_button.pack(pady=0)  # 复选框下方留有一点间距
show_axis = tk.BooleanVar()  # 用于存储复选框的值
check_button1 = tk.Checkbutton(f1, text="启用主轴", variable=show_axis)
check_button1.pack(pady=0)
var2 = tk.BooleanVar()  # 用于存储复选框的值
check_button2 = tk.Checkbutton(f1, text="选项 3", variable=var2)
check_button2.pack(pady=0)
var3 = tk.BooleanVar()  # 用于存储复选框的值
check_button3 = tk.Checkbutton(f1, text="选项 4", variable=var3)
check_button3.pack(pady=0)
var4 = tk.BooleanVar()  # 用于存储复选框的值
check_button4 = tk.Checkbutton(f1, text="选项 5", variable=var4)
check_button4.pack(pady=0)

label_of_index = tk.Label(f1,
                          width=6, height=2,
                          relief='flat',
                          command=None,
                          text='')
label_of_index.pack()

label_of_number = tk.Label(f2,
                          width=30, height=12,
                          relief='flat',
                          command=None,
                          text='')
label_of_number.pack()

def update_text():
    # 更新文本内容
    # 每1000毫秒（1秒）调用一次update_text函数
    root.after(1000, update_text)

content = control_ls()
open_dir()
root.mainloop()

