#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 23.7.21
"""

import os
import random
import sys

import cv2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from detect_image import ImgDetector
from myutils.cv_utils import generate_colors, draw_box_list
from myutils.make_html_page import make_html_page
from myutils.project_utils import traverse_dir_files, get_current_time_str
from root_dir import DATA_DIR


class ModelComparison(object):
    """
    模型对比
    """
    def __init__(self):
        self.weights1 = os.path.join(DATA_DIR, 'models', 'best-3c-20210715.pt')
        self.weights2 = os.path.join(DATA_DIR, 'models', 'best-3c-20210722.pt')
        self.predict1 = ImgDetector(self.weights1)
        self.predict2 = ImgDetector(self.weights2)

    @staticmethod
    def draw_clz_boxes(img_bgr, clz_dict):
        """
        绘制类别boxes
        """
        keys = list(clz_dict.keys())
        if len(keys) <= 3:
            color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        else:
            color_list = generate_colors(len(keys), 47)

        # 绘制boxes
        for idx, clz in enumerate(keys):
            box_list = clz_dict[clz]
            clz_idx = int(clz)
            img_bgr = draw_box_list(img_bgr, thickness=3, color=color_list[clz_idx], is_text=False,
                                    is_new=False, box_list=box_list)
        return img_bgr

    @staticmethod
    def save_img_patch(img_bgr, img_name):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        oss_root_dir = "zhengsheng.wcl/Character-Detection/datasets/image_tmps/"
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    def compare_path(self, img_path):
        """
        对比路径
        """
        print('[Info] img_path: {}'.format(img_path))
        img_bgr = cv2.imread(img_path)
        img_name_x = img_path.split("/")[-1].split(".")[0]
        clz_dict1 = self.predict1.detect_image(img_bgr)
        clz_dict2 = self.predict2.detect_image(img_bgr)
        img_out1 = ModelComparison.draw_clz_boxes(img_bgr, clz_dict1)
        img_out2 = ModelComparison.draw_clz_boxes(img_bgr, clz_dict2)
        img_name_out1 = "{}_v2.jpg".format(img_name_x)  # 旧版本
        img_name_out2 = "{}_v3.jpg".format(img_name_x)  # 新版本
        img_url_out1 = ModelComparison.save_img_patch(img_out1, img_name_out1)
        img_url_out2 = ModelComparison.save_img_patch(img_out2, img_name_out2)
        return [img_url_out1, img_url_out2]

    def compare(self):
        """
        对比模型
        """
        # img_dir = os.path.join(DATA_DIR, 'images')
        img_dir = os.path.join(DATA_DIR, 'ds_en_words_v3/images/val/')
        print('[Info] 对比文件夹: {}'.format(img_dir))
        num = 200
        print('[Info] num: {}'.format(num))

        html_file = os.path.join(DATA_DIR, 'data_v2_v3.{}.html'.format(get_current_time_str()))
        paths_list, _ = traverse_dir_files(img_dir)
        print('[Info] 文件数: {}'.format(len(paths_list)))

        if len(paths_list) > num:
            random.seed(47)
            paths_list = random.shuffle(paths_list)
            paths_list = paths_list[:num]

        print('[Info] 文件数: {}'.format(len(paths_list)))
        img_data_list = []
        for img_idx, img_path in enumerate(paths_list):
            try:
                img_data = self.compare_path(img_path)
            except Exception as e:
                continue
            img_data_list.append(img_data)
            print('[Info] img_idx: {}'.format(img_idx))
        make_html_page(html_file, img_data_list)  # 生成html
        print('[Info] 输出文件结果: {}'.format(html_file))


def main():
    mc = ModelComparison()
    mc.compare()


if __name__ == '__main__':
    main()
