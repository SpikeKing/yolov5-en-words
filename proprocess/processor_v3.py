#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 20.7.21
"""

from myutils.project_utils import *
from root_dir import DATA_DIR


class ProcessorV3(object):
    def __init__(self):
        self.file1_name = os.path.join(DATA_DIR, 'en_full.json')
        self.file2_name = os.path.join(DATA_DIR, '50url.txt')
        self.out_name = os.path.join(DATA_DIR, '50url.m.txt')

        # self.angle_err = os.path.join(DATA_DIR, 'en_full_with_alter.anno-angle.txt')

    def process(self):
        print("[Info] 处理数据")
        data1_lines = read_file(self.file1_name)
        print('[Info] 样本1 数量: {}'.format(len(data1_lines)))
        data1_dict = dict()
        for data1_line in data1_lines:
            data_dict = json.loads(data1_line)
            image_url = data_dict["image_url"]
            image_name_x = image_url.split("/")[-1].split(".")[0]
            data1_dict[image_name_x] = image_url
        print('[Info] 1 数据量: {}'.format(len(data1_dict.keys())))

        data2_lines = read_file(self.file2_name)
        print('[Info] 样本2 数量: {}'.format(len(data2_lines)))
        for data2_line in data2_lines:
            data_dict = json.loads(data2_line)
            image_url = data_dict["image_url"]
            image_name_x = image_url.split("/")[-1].split(".")[0]
            # image_name_x, angle = image_name_x.split("_")
            image_original_url = data1_dict[image_name_x]
            # if angle != "0":
            #     continue
            data_dict["image_url"] = image_original_url
            write_line(self.out_name, json.dumps(data_dict))

        print('[Info] 处理完成!')

    def process_v2(self):
        print("[Info] 处理数据")
        data1_lines = read_file(self.file1_name)
        print('[Info] 样本1 数量: {}'.format(len(data1_lines)))
        data1_dict = dict()
        for data1_line in data1_lines:
            data_dict = json.loads(data1_line)
            image_url = data_dict["image_url"]
            image_name_x = image_url.split("/")[-1].split(".")[0]
            data1_dict[image_name_x] = image_url
        print('[Info] 1 数据量: {}'.format(len(data1_dict.keys())))

        data2_lines = read_file(self.file2_name)
        print('[Info] 样本2 数量: {}'.format(len(data2_lines)))
        for data2_line in data2_lines:
            if not data2_line:
                continue
            image_url = data2_line
            image_name_x = image_url.split("/")[-1].split(".")[0]
            image_name_x, angle = image_name_x.split("_")
            if image_name_x not in data1_dict.keys():
                print('[Info] image_name_x: {}'.format(image_name_x))
                continue
            image_original_url = data1_dict[image_name_x]
            write_line(self.out_name, image_original_url)
        print('[Info] 处理完成!')


def main():
    prc3 = ProcessorV3()
    prc3.process_v2()


if __name__ == '__main__':
    main()
