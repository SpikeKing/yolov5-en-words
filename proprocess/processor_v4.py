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
        self.file1_name = os.path.join(DATA_DIR, 'en_lower.json')

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
            image_name_x, angle = image_name_x.split("_")
            image_original_url = data1_dict[image_name_x]
            if angle != "0":
                continue
                # write_line(self.angle_err, image_original_url)
            data_dict["image_url"] = image_original_url
            write_line(self.out_name, json.dumps(data_dict))

        print('[Info] 处理完成!')


def main():
    prc3 = ProcessorV3()
    prc3.process()


if __name__ == '__main__':
    main()