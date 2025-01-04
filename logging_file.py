# logging_file.py
import logging
import os
from datetime import datetime

# �����Ƿ�������־��¼�Ŀ���
enable_logging = True  # ����Ϊ False �ɽ�����־��¼


def setup_logging():
    # ���������־��¼��ֱ�ӷ��أ������κ�����
    if not enable_logging:
        return None

    # ��ȡ��ǰʱ�������Ϊ��־�ļ���
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")

    # ������־�ļ��洢Ŀ¼
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # ���Ŀ¼�����ڣ�����Ŀ¼

    log_filepath = os.path.join(log_dir, log_filename)

    # ������־��¼��
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # ������־����

    # �����ļ����������������ļ���ʽ
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # ��������̨�������������ÿ���̨�����ʽ
    console_handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # ����������ӵ���־��¼��
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger  # ������־��¼��


# �����Ҫ�ڳ�������ʱ�ͳ�ʼ����־������ֱ�ӵ��� setup_logging
logger = setup_logging()