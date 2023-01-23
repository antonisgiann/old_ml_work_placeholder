import logging
import os
import sys
from datetime import datetime

def get_my_logger(file_path):
    """
    Just a simple logger
    """
    running_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    ROOT_DIR = os.path.dirname(file_path)
    if not os.path.exists(f"{ROOT_DIR}/log"):
        os.makedirs(f"{ROOT_DIR}/log")

    # Logging
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S"
    )

    fileHandler = logging.FileHandler(filename=f"{ROOT_DIR}/log/{running_timestamp}.log")
    fileHandler.setFormatter(formatter)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger = logging.getLogger(os.path.basename(file_path))

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.DEBUG)
    
    return logger
