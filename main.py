import csv
import tkinter as tk
from tkinter import filedialog

import numpy as np

import core
from PIL import Image, ImageTk
import os

from matplotlib import rc

from logging_file import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# 避免中文乱码
rc("font", family='YouYuan')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# model = 'runs/segment/train10 extend/weights/best.pt'
model_path = 'model/base.pt'
default_picdir = r'C:\Users\Mr zhang\Desktop\张霄\2023.0817\10X反'


ls_of_photo = []
ls_of_time = []
dir_of_result = dict()

index = 0

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
        self.cache_window = 5  # 缓存的前后图片范围
        self.cache_thread = None  # 后台线程
        self.lock = threading.Lock()  # 用于线程同步

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
        self.lock.acquire()
        results = Model.return_results(path)
        result = results[0]
        start_time = time.time()
        new_pic = dealt_pic_cache(result)
        end_time = time.time()
        self.ls_of_time.append(path)
        self.deal_pics_cache[path] = new_pic
        logger.debug(f"add_new_pic运行时间: {end_time - start_time} 秒")
        self.lock.release()

    def remove_pic(self):
        pass

    def garbage_collection(self):
        pass
    
    def save_data_in_thread(self):
        folder_path = tk.filedialog.askdirectory()
        update_status(f"{folder_path}已经加入处理队列")
        if folder_path:
            pass
        else:
            return None
        thread = threading.Thread(target=self.save_all_data_as_csv,args=(folder_path,))
        thread.start()

    def save_all_data_as_csv(self,folder_path):
        def return_PCA_result_list(dir_path, filename, mod):
            '''
            单张图片的
            :param dir_path:file location
            :param filename:str
            :param mod: mode
            :return:list
            '''
            global ls
            try:
                filepath = os.path.join(dir_path, filename)
                c = mod.return_results(filepath)
                i = c[0]
                ls = []
                for j in i.return_cell():
                    bit = j.return_bit()
                    j.analyse_type2(bit)
                    # 气孔面积  保卫细胞面积  是否打开 保卫细胞方向x 保卫细胞方向y 气孔方向x 气孔方向y 面积占比 最大长度 最大宽度 属于的叶片（文件路径）,文件夹','文件名'
                    data_set = (
                        j.cell_area,
                        j.contour,
                        j.cell_area / j.contour,
                        j.params['conf'],
                        j.cell_PCA.main[0, 0] if j.cell_PCA.main[0, 0] else None,
                        j.cell_PCA.main[0, 1] if j.cell_PCA.main[0, 1] else None,
                        j.cell_PCA.max_axis_length if j.cell_PCA.max_axis_length else None,
                        j.cell_PCA.max_wide if j.cell_PCA.max_wide else None,
                        filepath,
                        dir_path,
                        os.path.split(filepath)[1]
                    )
                    ls.append(data_set)
            except Exception as e:
                print(e)
            return ls

        # if self.dir_path is None:
        #     folder_path = tk.filedialog.askdirectory()
        # else:
        #     folder_path = self.dir_path
        self.lock.acquire()
        name = os.path.split(folder_path)[1]
        with open(name + 'result.csv', "w", encoding='utf8', newline='') as f:
            a = csv.writer(f, )
            a.writerow(('气孔面积', '气孔周长', '形状指数', '置信度', '气孔方向x', '气孔方向y', '最大长度','最大宽度','属于的叶片（文件路径）','文件夹','文件名',
                        '叶子序数', '气孔序数'))
        n_of_leaf = 0
        ls_of_pic = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 获取每个文件的完整路径
                if file.endswith(('.bmp', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    ls_of_pic.append(file_path)

        for n,i in enumerate(ls_of_pic):
            ls = return_PCA_result_list(folder_path, i, Model)
            print(f"保存第{n+1}中,共{len(ls_of_pic)}个")
            update_status(f"保存{name}第{n+1}中,共{len(ls_of_pic)}个")
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
        update_status(f"保存在{name + 'result.csv'}中")
        self.lock.release()


    def start_cache_thread(self):
        """启动后台线程，定时检查并更新缓存"""
        if self.cache_thread is None or not self.cache_thread.is_alive():
            print('线程启动')
            self.cache_thread = threading.Thread(target=self.cache_updater)
            self.cache_thread.daemon = True
            self.cache_thread.start()

    def cache_updater(self):
        """后台线程，每隔一段时间检查并更新缓存"""
        while True:
            print('trying,now have {} caches'.format(len(self.deal_pics_cache)))
            self.load_cache_for_index()  # 每次检查并更新当前索引前后3张图像的缓存
            time.sleep(2)  # 每隔2秒检查一次，可以根据需求调整时间间隔

    def load_cache_for_index(self):
        """加载当前 index 前后缓存的图片"""
        start_index = max(0, self.index - self.cache_window)
        end_index = min(len(self.ls_of_photo), self.index + self.cache_window + 1)
        print("将加载第{}-{}张图片".format(start_index, end_index))
        for i in range(start_index, end_index):
            path = self.ls_of_photo[i]
            if path not in self.deal_pics_cache:
                self.add_new_pic(path)
                print("加载图片中 {},目前有{}个缓存".format(path,len(self.deal_pics_cache)))# 加载并缓存图片

content = control_ls()


class dealt_pic_cache():
    '''
    处理过的类
    '''

    def __init__(self, orig: core.Result):
        # 检查传入参数orig的有效性
        if not isinstance(orig, core.Result):
            raise TypeError("Expected 'core.Result' type for 'orig', got {}".format(type(orig)))
        self.source = orig
        self.pic = orig.pic

        # 初始化默认值
        self.a = 0.5  # 透明度默认值

        # 设置默认颜色蒙版
        color_mask = np.zeros_like(self.pic,dtype=np.uint8)
        color_mask[:, :, 0] = 240  # R
        color_mask[:, :, 1] = 140  # G
        color_mask[:, :, 2] = 50  # B

        # 设置默认轴线颜色
        axis_color = [150, 50, 240]
        axis_layer = np.zeros_like(self.pic)

        # 初始化面积和周长的累计值
        total_area = 0
        total_perimeter = 0

        # 获取细胞列表
        cell_list = orig.return_cell()

        # 如果细胞列表为空，我们不抛出异常，而是设置属性为无效值
        if not cell_list:
            self.mean_area = -1
            self.mean_axis = -1
            self.s_counter = -1
            self.mask = np.zeros_like(self.pic)  # 空的mask
            self.masked_pic = np.zeros_like(self.pic)  # 空的处理后的图像
            return  # 结束构造函数，不进行后续的计算

        # 创建并计算掩码层和轴线层
        mask_layer = np.zeros(self.source.orig_shape,dtype=np.uint8)
        axis_layer = np.zeros_like(self.pic,dtype=np.uint8)
        start_time = time.time()
        for i in cell_list:
            # 累加细胞的掩码
            bit = i.return_bit()
            mask_layer += bit

            # 分析细胞，计算面积和周长
            start_time3 = time.time()
            i.analyse_type2(bit)
            end_time4 = time.time()
            logger.debug(f"分析2的运行时间: {end_time4 - start_time3} 秒")

            total_area += i.cell_area
            total_perimeter += i.contour
            end_time = time.time()

            # 更新轴线数据
            if i.cell_PCA is not None:
                y = np.clip(i.cell_PCA.line_y.astype(np.int32), 0, self.source.orig_shape[0] - 1)
                x = np.clip(i.cell_PCA.line_x.astype(np.int32), 0, self.source.orig_shape[1] - 1)
                axis_layer[y, x] = axis_color

            end_time3 = time.time()
            logger.debug(f"处理一张图片细胞的运行时间: {end_time3 - start_time3} 秒")

        end_time2 = time.time()
        logger.debug(f"deal_pic中PCA运行时间: {end_time2 - start_time} 秒")
        # 确保mask_layer值在[0, 1]之间，防止溢出
        mask_layer = np.clip(mask_layer, 0, 1)
        mask_layer = np.expand_dims(mask_layer, axis=2)  # 增加维度以便与颜色蒙版合并
        mask_layer = np.repeat(mask_layer, 3, axis=2)  # 重复三次形成RGB
        color_mask *= mask_layer  # 将颜色蒙版与掩码叠加
        mask_layer = np.where(mask_layer == 1, self.a, 1)  # 根据透明度调整mask

        # 最终掩码图像
        self.axis_layer = axis_layer
        self.res_in_mask = mask_layer * self.pic + color_mask * (1 - self.a)
        # 计算气孔的平均面积和平均周长

        avg_area = total_area / len(cell_list)
        avg_perimeter = total_perimeter / len(cell_list)

        # 更新气孔信息
        self.mean_area = avg_area
        self.mean_axis = avg_perimeter
        self.s_counter = len(cell_list)
        self.c_counter = sum([1 for i in cell_list if i.params["conf"] > 0.5]) #置信度
        logger.debug(f"deal_pic运行时间: {end_time - start_time} 秒")
        del self.source
    def update_result(self,orig):
        # mask_layer = np.zeros(orig.orig_shape)
        # for i in self.mask_ls:
        #     mask_layer += i
        # self.masked_pic = mask_layer
        pass

    def return_pic(self):
        res = self.pic
        if show_mask.get():
            res = self.res_in_mask
        if show_axis.get():
            res[self.axis_layer != 0] = self.axis_layer[self.axis_layer != 0]
        return res


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

    def get_pic(self):
        pass


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
    start_time = time.time()
    if content.is_show == 0:
        # 如果不需要处理，直接显示图片
        img = img_deal(path)
        main_img.itemconfig(image_id, image=img)
        return

    # 检查缓存中是否已有处理过的图像
    if path not in content.deal_pics_cache:
        # 如果缓存中没有，处理该图像并生成一个新的缓存实例
        content.add_new_pic(path)
    else:
        print("检测到缓存")
    end_time2 = time.time()

     # 创建类实例并缓存
    # 获取缓存中的处理结果
    cached_result = content.deal_pics_cache[path]
    # 从缓存实例中获取处理后的图像
    res = cached_result.return_pic().astype('uint8')

    # 更新图片显示
    img = img_deal(res)
    main_img.itemconfig(image_id,image=img)
    # 获取并显示气孔的统计信息
    show_text = f'''
        气孔数量: {cached_result.s_counter}
        高于置信度的气孔数量: {cached_result.c_counter}
        平均面积: {cached_result.mean_area if cached_result.mean_area is not None else 0:.2f}
        平均周长: {cached_result.mean_axis if cached_result.mean_axis is not None else 0:.2f}
    '''
    label_of_number.config(text=show_text)
    end_time = time.time()
    logger.debug(f"show运行时间: {end_time - start_time:3f} 秒")
    logger.debug(f"show中add_new_pic运行时间: {end_time2 - start_time:3f} 秒")


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
        if os.path.exists(origin):
            folder_path = origin
        else:
            folder_path = tk.filedialog.askdirectory()
        content.first_open = False
    else:
        folder_path = tk.filedialog.askdirectory()
    if folder_path:
        # 清空列表
        content.clear()
        content.get_name(folder_path)
    else:
        return None
    start_time = time.time()
    if len(content.ls_of_photo) > 0:
        show(content.ls_of_photo[0])
        content.index = 0
        change_show_index()
        # for i in content.ls_of_photo:
        #     result = Model.return_results(i)
        #     content.ls_of_time.append(i)
        #     content.deal_pics_cache[i] = dealt_pic_cache(result[0])
        #     break
    else:
        print("没有检测到文件夹中的照片")
    end_time = time.time()
    logger.debug(f"open_dir运行时间: {end_time - start_time} 秒")

def change_show_index():
    total = len(content.ls_of_photo)
    t = content.index + 1
    label_of_index.config(text="图片数量\n{}--{}".format(t, total))


def img_deal(input, canvas_width=1024, canvas_height=768):
    """
    处理输入图像（路径或numpy数组），并按比例缩放图像，使其适配Canvas。

    参数:
    - input: 图像文件路径 (str) 或 numpy 数组 (np.ndarray)
    - canvas_width: Canvas宽度，默认为800
    - canvas_height: Canvas高度，默认为600

    返回:
    - img2_avoid_trash: Tkinter 可用的图像对象
    """
    # 根据输入类型加载图像
    if isinstance(input, str):  # 如果输入是路径，加载图像文件
        img = Image.open(input)
    elif isinstance(input, np.ndarray):  # 如果输入是numpy数组
        img = Image.fromarray(input.astype(np.uint8))  # 将numpy数组转换为PIL图像
    else:
        print(f"Unsupported input type: {type(input)}")
        return None

    global img2_avoid_trash  # 防止图像被垃圾回收
    original_width, original_height = img.size

    # 计算图像缩放比例，使图像完全适配Canvas
    scale_x = canvas_width / original_width
    scale_y = canvas_height / original_height

    # 选择较小的比例，确保图像完全适应Canvas
    scale = min(scale_x, scale_y)

    # 计算新的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 使用 Pillow 缩放图像
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    # 将PIL图像转换为Tkinter可显示的格式
    img2_avoid_trash = ImageTk.PhotoImage(img_resized)

    return img2_avoid_trash

def left_chick_of_img(event):
    pass


windows_width = 1350
windows_height = 768

root = tk.Tk()
root.geometry(str(windows_width) + "x" + str(windows_height) + "+0+0")
root.title("气孔识别")
main_img = tk.Canvas(root,
                    bg='black',
                    height=768, width=1024,
                    )
image_id = main_img.create_image( 1024/2,768/2, anchor="center", image=None)
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

open_dir_button.pack()
show_button = tk.Button(f1,
                        width=6, height=2,
                        relief='raise',
                        command=content.save_data_in_thread,
                        text='保存为csv')
show_button.pack()

status_frame = tk.Frame(root,
                       width=windows_width, 
                       height=15,
                       borderwidth=0,  # 移除边框
                       bd=0,           # 移除边框
                       bg='#2B2B2B',
                       relief='flat')  # 移除突出效果
status_frame.place(relx=0, 
                  rely=1.0,
                  relwidth=1.0,
                  anchor='sw',
                  height=15,
                  x=0,    # 移除左边距
                  y=0)    # 移除底部边距，确保完全贴底

# 调整文本框样式
status_text = tk.Text(status_frame,
                     width=50,
                     height=1,
                     wrap=tk.WORD,
                     bg='#2B2B2B',
                     fg='#FFFFFF',
                     relief='flat',
                     padx=5,
                     pady=0,
                     border=0,
                     highlightthickness=0)
status_text.place(relx=0.99,
                 rely=0.5,
                 anchor='e',
                 relwidth=0.3,
                 height=13)

def update_status(message):
    status_text.delete(1.0, tk.END)
    status_text.insert(tk.END, f"{message}")
    status_text.see(tk.END)

# 在 root.mainloop() 之前添加一些初始状态信息
update_status("程序启动完成")



open_dir()
content.start_cache_thread()
root.mainloop()

