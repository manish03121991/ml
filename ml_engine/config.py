import os
import logging
from logging.handlers import RotatingFileHandler

# Get ml_engine folder path
BASE_DIR_PATH = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

# set pickle file path
MODEL_FILES_PATH = os.path.join('ml_engine', 'pkl/', 'classification/')

# set capture for model training filename
MODEL_FILE_CAPTURE = 'MODEL'

# set logger
LOG_PATH = os.path.join(BASE_DIR_PATH, 'ml_engine', 'logs/')
ML_LOG_FILE = LOG_PATH + 'ml_log.log'
ml_logger = logging.getLogger('ml_logger')
ml_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler(
    ML_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
handler.setFormatter(formatter)
ml_logger.addHandler(handler)

# set logger for model
MODEL_LOG_FILE = LOG_PATH + 'model.log'
model_logger = logging.getLogger('model_logger')
model_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler(
    MODEL_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
handler.setFormatter(formatter)
model_logger.addHandler(handler)
