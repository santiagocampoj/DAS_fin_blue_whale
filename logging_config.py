import logging

def setup_logging(log_file='DAS.log', level=logging.DEBUG):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create a file handler that overwrites the log file each time
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode overwrites the file

    file_formatter = logging.Formatter('%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_formatter)

    # check if the logger already has handlers, and if so, clear them
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)

    return logger
