#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 22.7.21
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing.pool import Pool
from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class DataProcessorV2(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "hw_data_v1_20210722.txt")
        self.out_path = os.path.join(
            DATA_DIR, "hw_data_v1_20210722.out-{}.txt".format(get_current_time_str()))

    def list_2_rec(self, anno_box_str):
        """
        list转换为rec
        """
        a_list = [int(float(x)) for x in anno_box_str.split(",")]
        n = len(a_list)
        rec = []
        for i in range(0, n, 2):
            rec.append([a_list[i], a_list[i+1]])
        return rec

    @staticmethod
    def save_img_patch(img_bgr, img_name):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        oss_root_dir = "zhengsheng.wcl/Character-Detection/datasets/hw_words_dataset_20210722/"
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def filter_boxes(hw_box, rec_list):
        """
        过滤boxes
        """
        box_list = []
        for rec in rec_list:
            rec = np.array(rec).astype(np.float32)
            x, y, w, h = cv2.boundingRect(rec)
            box = [x, y, x + w, y + h]
            v_iou = min_iou(hw_box, box)
            if v_iou > 0.6:
                box[0] = max(box[0] - hw_box[0], 0)
                box[1] = max(box[1] - hw_box[1], 0)
                box[2] = min(box[2] - hw_box[0], hw_box[2])
                box[3] = min(box[3] - hw_box[1], hw_box[3])
                box_list.append(box)
        return box_list

    @staticmethod
    def process_img(img_idx, img_url, img_data_dict, out_path):
        img_name = img_url.split("/")[-1].split(".")[0]
        items = img_data_dict[img_url]
        hw_line_list, english_list, chinese_list, tugai_list = items
        for hw_idx, hw_rec in enumerate(hw_line_list):
            hw_rec = np.array(hw_rec).astype(np.float32)
            x, y, w, h = cv2.boundingRect(hw_rec)
            hw_box = [x, y, x + w, y + h]

            res_english_list = DataProcessorV2.filter_boxes(hw_box, english_list)
            res_chinese_list = DataProcessorV2.filter_boxes(hw_box, chinese_list)
            res_tugai_list = DataProcessorV2.filter_boxes(hw_box, tugai_list)
            _, img_bgr = download_url_img(img_url)
            # print('[Info] img_bgr: {}'.format(img_bgr.shape))
            img_patch = get_cropped_patch(img_bgr, hw_box)
            img_patch_name = "{}_{}.jpg".format(img_name, str(hw_idx))
            img_patch_url = DataProcessorV2.save_img_patch(img_patch, img_patch_name)

            # draw_box_list(img_patch, res_tugai_list, save_name="tmp2.jpg")
            data_dict = {
                "img_patch_url": img_patch_url,
                "english_list": res_english_list,
                "chinese_list": res_chinese_list,
                "tugai_list": res_tugai_list
            }
            write_line(out_path, json.dumps(data_dict))
        print('[Info] 处理完成: {}'.format(img_idx))

    def process(self):
        data_lines = read_file(self.file_path)
        print("[Info] 处理开始: {}".format(self.file_path))
        img_dict = collections.defaultdict(list)
        for data_line in data_lines:
            items = data_line.split("<sep>")
            img_url = items[0]
            anno_tag = items[1]
            anno_box_str = items[2]
            anno_rec = self.list_2_rec(anno_box_str)
            img_dict[img_url].append((anno_tag, anno_rec))
        print('[Info] 图像数: {}'.format(len(img_dict.keys())))

        img_data_dict = collections.defaultdict(list)
        for img_url in img_dict.keys():
            data_list = img_dict[img_url]
            hw_line_list, english_list, chinese_list, tugai_list = [], [], [], []
            for data in data_list:
                anno_tag, anno_rec = data
                if anno_tag == "WenBenHang":
                    hw_line_list.append(anno_rec)
                elif anno_tag == "YingYuDanCi":
                    english_list.append(anno_rec)
                elif anno_tag == "ZhongWenShouXie":
                    chinese_list.append(anno_rec)
                elif anno_tag == "TuGai":
                    tugai_list.append(anno_rec)
            img_data_dict[img_url] = [hw_line_list, english_list, chinese_list, tugai_list]
        print('[Info] 图像数: {}'.format(len(img_data_dict.keys())))

        pool = Pool(processes=100)
        for img_idx, img_url in enumerate(img_data_dict.keys()):
            pool.apply_async(DataProcessorV2.process_img, (img_idx, img_url, img_data_dict, self.out_path))
        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(self.out_path))

def main():
    dp2 = DataProcessorV2()
    dp2.process()


if __name__ == '__main__':
    main()
