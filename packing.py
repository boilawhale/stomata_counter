from cx_Freeze import setup, Executable
from main import model_path
import os

# 指定需要包含的外部文件，包括 DLL
my_path = r"D:\anaconda3\envs\yolo8"
build_exe_options = {
    "packages": ["os", "sys", "numpy", "scipy"],
    "include_files": [
        (os.path.join(my_path,"vcomp140.dll"),'vcomp140.dll'), # 手动添加 DLL 文件
        (os.path.join(my_path,"msvcp140.dll"),'msvcp140.dll'),
        (model_path,'model/base.pt'),
    ],
}

setup(
    name="气孔识别",
    version="1.0",
    description="sample program",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py")],
)

