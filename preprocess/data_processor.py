#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 8.7.21
"""

import cv2
import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class DataProcessor(object):
    """
    英语单词数据处理
    """
    def __init__(self):
        # self.file_name = os.path.join(DATA_DIR, 'word_annotations.20210714153600.txt')
        self.file_name = os.path.join(DATA_DIR, 'hw_data_v1_20210722.out-20210722173343.txt')
        # self.file_name = os.path.join(DATA_DIR, 'hw_data_v2_20210722.out-20210722175706.txt')
        # self.out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'ds_en_words_v1')
        self.out_dir = os.path.join(DATA_DIR, 'ds_en_words_v3')
        mkdir_if_not_exist(self.out_dir)
        self.imgs_dir = os.path.join(self.out_dir, 'images')
        self.lbls_dir = os.path.join(self.out_dir, 'labels')
        mkdir_if_not_exist(self.imgs_dir)
        mkdir_if_not_exist(self.lbls_dir)
        self.train_imgs_dir = os.path.join(self.imgs_dir, 'train')
        self.val_imgs_dir = os.path.join(self.imgs_dir, 'val')
        mkdir_if_not_exist(self.train_imgs_dir)
        mkdir_if_not_exist(self.val_imgs_dir)
        self.train_lbls_dir = os.path.join(self.lbls_dir, 'train')
        self.val_lbls_dir = os.path.join(self.lbls_dir, 'val')
        mkdir_if_not_exist(self.train_lbls_dir)
        mkdir_if_not_exist(self.val_lbls_dir)

    @staticmethod
    def convert(iw, ih, box):
        """
        将标注的xml文件标注转换为darknet形的坐标
        """
        iw = float(iw)
        ih = float(ih)
        dw = 1. / iw
        dh = 1. / ih
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def process_line(idx, data_line, imgs_dir, lbls_dir):
        # print('[Info] idx: {}'.format(idx))
        data_dict = json.loads(data_line)
        # img_url = data_dict['image_url']
        # english_bbox_list = data_dict['english_bbox_list']
        # chinese_bbox_list = data_dict['chinese_bbox_list']
        # alter_bbox_list = data_dict['alter_bbox_list']
        img_url = data_dict['img_patch_url']
        english_bbox_list = data_dict['english_list']
        chinese_bbox_list = data_dict['chinese_list']
        alter_bbox_list = data_dict['tugai_list']

        # 不同文件使用不同的文件名
        file_idx = str(idx).zfill(5)
        img_path = os.path.join(imgs_dir, 'v3_{}.jpg'.format(file_idx))
        lbl_path = os.path.join(lbls_dir, 'v3_{}.txt'.format(file_idx))

        # 写入图像
        is_ok, img_bgr = download_url_img(img_url)
        cv2.imwrite(img_path, img_bgr)  # 写入图像

        # 写入标签
        ih, iw, _ = img_bgr.shape  # 高和宽
        res_bboxes_lines = []

        # 写入3个不同标签
        for bbox in english_bbox_list:
            bbox_yolo = DataProcessor.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            res_bboxes_lines.append(" ".join(["0", *bbox_yolo]))
        for bbox in chinese_bbox_list:
            bbox_yolo = DataProcessor.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            res_bboxes_lines.append(" ".join(["1", *bbox_yolo]))
        for bbox in alter_bbox_list:
            bbox_yolo = DataProcessor.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            res_bboxes_lines.append(" ".join(["2", *bbox_yolo]))

        create_file(lbl_path)
        write_list_to_file(lbl_path, res_bboxes_lines)
        print('[Info] idx: {} 处理完成: {}'.format(idx, img_path))

    def process(self):
        print('[Info] 处理数据: {}'.format(self.file_name))
        data_lines = read_file(self.file_name)
        n_lines = len(data_lines)
        # data_lines = data_lines[:20]  # 测试
        random.seed(47)
        random.shuffle(data_lines)
        print('[Info] 文件数: {}'.format(n_lines))

        n_x = 20
        n_split = len(data_lines) // n_x
        train_lines = data_lines[:n_split*(n_x-1)]
        val_lines = data_lines[n_split*(n_x-1):]
        print('[Info] 训练: {}, 测试: {}'.format(len(train_lines), len(val_lines)))

        pool = Pool(processes=100)

        for idx, data_line in enumerate(train_lines):
            # DataProcessor.process_line(idx, data_line, self.train_imgs_dir, self.train_lbls_dir)
            pool.apply_async(DataProcessor.process_line,
                             (idx, data_line, self.train_imgs_dir, self.train_lbls_dir))

        for idx, data_line in enumerate(val_lines):
            # DataProcessor.process_line(idx, data_line, self.val_imgs_dir, self.val_lbls_dir)
            pool.apply_async(DataProcessor.process_line,
                             (idx, data_line, self.val_imgs_dir, self.val_lbls_dir))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_dir))


def main():
    dp = DataProcessor()
    dp.process()


if __name__ == '__main__':
    main()
