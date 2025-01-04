# logging_file.py
import logging
import os
from datetime import datetime

# 配置是否启用日志记录的开关
enable_logging = True  # 设置为 False 可禁用日志记录


def setup_logging():
    # 如果禁用日志记录，直接返回，不做任何配置
    if not enable_logging:
        return None

    # 获取当前时间戳，作为日志文件名
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")

    # 设置日志文件存储目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 如果目录不存在，创建目录

    log_filepath = os.path.join(log_dir, log_filename)

    # 创建日志记录器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建文件处理器，并设置文件格式
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 创建控制台处理器，并设置控制台输出格式
    console_handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger  # 返回日志记录器


# 如果需要在程序启动时就初始化日志，可以直接调用 setup_logging
logger = setup_logging()