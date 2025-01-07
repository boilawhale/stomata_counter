# logging_file.py
import logging
import os
from datetime import datetime

# 控制日志记录功能的全局开关
enable_logging = True  # 设置为 False 可以完全禁用日志功能


def setup_logging():
    """
    配置并初始化日志系统
    返回: logging.Logger 对象，如果日志功能被禁用则返回 None
    """
    # 如果日志功能被禁用，直接返回 None
    if not enable_logging:
        return None

    # 使用当前时间戳生成唯一的日志文件名
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")

    # 创建日志文件存储目录（如果不存在）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filepath = os.path.join(log_dir, log_filename)

    # 创建并配置主日志记录器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置为最详细的日志级别

    # 配置文件处理器
    # 将日志消息写入文件，包含所有级别的日志
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 配置控制台处理器
    # 只显示 INFO 及以上级别的日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建全局日志记录器实例
logger = setup_logging()