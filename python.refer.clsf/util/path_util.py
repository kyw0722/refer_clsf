 # coding:utf-8
import os

PROJECT_NAME = 'python.refer.clsf'     # PROJECT 이름에 맞게 수정

def get_project_root_path(project_name=PROJECT_NAME):
    current_dir = os.getcwd()
    current_dir_list = current_dir.split(os.sep)

    for idx, dir_name in enumerate(current_dir_list):
        if project_name == dir_name: break

    return os.sep.join(current_dir_list[:idx + 1])




