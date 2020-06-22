import os
import logging
import sys
import time


def assert_exists(filepath):
    assert os.path.exists(filepath), f'{filepath} not found'


def build_logger(name, log_file):
    log_formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(process)d %(levelname)s %(module)s:%(lineno)s - %(funcName)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


def index_by_id(l):
    return {x['id']: x for x in l}