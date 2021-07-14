#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.6.21
"""

import os
import sys

import xml.dom.minidom

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class SampleParser(object):
    """
    样本解析, 将样本转换为简易格式
    """
    def __init__(self):
        self.sample_path = os.path.join(DATA_DIR, 'word_annotations.20210708.xml')
        self.out_labeled = os.path.join(DATA_DIR, 'word_annotations.{}.txt'.format(get_current_time_str()))

    @staticmethod
    def split_boxes(pnt_list, box):
        """
        根据点列表拆分box
        """
        if not pnt_list:
            return [box]
        x_min, y_min, x_max, y_max = box
        x_list = []
        for pnt in pnt_list:
            x_list.append(pnt[0])
        x_list = sorted(x_list)
        sub_boxes = []
        x_s = x_min
        for x in x_list:
            sub_boxes.append([x_s, y_min, x, y_max])
            x_s = x
        sub_boxes.append([x_s, y_min, x_max, y_max])
        return sub_boxes

    @staticmethod
    def parse_pnt_and_box(box_pnt_dict, box_list, img_bgr=None):
        """
        解析点和box
        """
        sub_boxes_list = []

        for idx in box_pnt_dict.keys():
            pnt_list = box_pnt_dict[idx]
            # print('[Info] pnt_list: {}'.format(pnt_list))
            box = box_list[idx]
            sub_boxes = SampleParser.split_boxes(pnt_list, box)
            sub_boxes_list.append(sub_boxes)

        sub_boxes_list = unfold_nested_list(sub_boxes_list)  # 双层list变成单层list

        # 划掉文字的区域需要区分对待
        for x_idx in range(len(box_list)):
            if x_idx not in box_pnt_dict.keys():
                sub_boxes_list.append(box_list[x_idx])

        # tmp_path = os.path.join(DATA_DIR, 'tmps', 'sub_boxes.jpg')
        # draw_box_list(img_bgr, sub_boxes_list, is_text=False, color=(255, 0, 0), save_name=tmp_path)
        return sub_boxes_list

    @staticmethod
    def parse_points(pnt_list_raw):
        """
        解析点
        """
        pnt_str_list = pnt_list_raw.split(";")
        pnt_list = []
        for pnt_str in pnt_str_list:
            p_list = pnt_str.split(',')
            pnt = [int(float(p)) for p in p_list]
            pnt_list.append(pnt)
        return pnt_list

    def process_annotations(self):
        """
        处理解析标签
        """
        url_format = "http://quark-cv-data.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Character-Detection/" \
                     "datasets/english-words-patch-20210702/{}"
        DOMTree = xml.dom.minidom.parse(self.sample_path)
        collection = DOMTree.documentElement
        meta = collection.getElementsByTagName("meta")
        # print('[Info] meta: {}'.format(meta))
        image_data = collection.getElementsByTagName("image")
        print('[Info] 样本数: {}'.format(len(image_data)))

        data_lines = []  # 标签信息列表
        for image in image_data:
            image_name = image.getAttribute("name")
            # print('[Info] image: {}'.format(image_name))
            image_url = url_format.format(image_name)
            # print('[Info] image_url: {}'.format(image_url))
            # _, img_bgr = download_url_img(image_url)
            # print('[Info] img_bgr: {}'.format(img_bgr.shape))
            polygon_data = image.getElementsByTagName("polygon")
            # print('[Info] polygon_data num: {}'.format(len(polygon_data)))
            english_bbox_list, chinese_bbox_list, alter_bbox_list = [], [], []
            for polygon in polygon_data:
                label = polygon.getAttribute("label")
                points = polygon.getAttribute("points")
                # print('[Info] label: {}, points: {}'.format(label, points))

                points = self.parse_points(points)
                # print('[Info] points: {}'.format(points))
                x, y, w, h = cv2.boundingRect(np.array(points))
                bbox = [x, y, x+w, y+h]
                if label == "YingYuDanCi":
                    english_bbox_list.append(bbox)
                elif label == "ZhongWenShouXie":
                    chinese_bbox_list.append(bbox)
                elif label == "TuGai":
                    alter_bbox_list.append(bbox)

            # print('[Info] bboxes数量: {}'.format(len(bbox_list)))
            # draw_box_list(img_bgr, bbox_list, is_show=True, save_name="tmp.jpg")
            data_dict = {
                "image_url": image_url,
                "english_bbox_list": english_bbox_list,
                "chinese_bbox_list": chinese_bbox_list,
                "alter_bbox_list": alter_bbox_list,
            }
            data_line = json.dumps(data_dict)
            data_lines.append(data_line)

        print('[Info] 行数: {}'.format(len(data_lines)))
        write_list_to_file(self.out_labeled, data_lines)
        print('[Info] 写入完成: {}'.format(self.out_labeled))


def main():
    slp = SampleParser()
    slp.process_annotations()


if __name__ == '__main__':
    main()
