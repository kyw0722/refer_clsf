# coding:utf-8
'''config 사용'''
import os
from util import path_util
import configparser

config = configparser.ConfigParser()
config_path = os.path.join(path_util.get_project_root_path(), 'config/config.ini')
config.read(config_path, encoding='utf-8')

