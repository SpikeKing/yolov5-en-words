#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 15.7.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from detect_image import ImgDetector
from root_dir import DATA_DIR

from myutils.project_utils import *
from myutils.cv_utils import *

class Processor(object):
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, 'en_full.json')
        self.out_path = os.path.join(DATA_DIR, 'en_full_with_alter.out-{}.txt'.format(get_current_time_str()))

    @staticmethod
    def parse_pos_2_rec(pos_data):
        """
        解析pos
        """
        pos_list = []
        for pos in pos_data:
            x = pos['x']
            y = pos['y']
            pos_list.append([x, y])
        return pos_list

    @staticmethod
    def filter_en_line(line_str):
        """
        过滤英文行
        """
        is_en = check_english_str(line_str)
        is_size = len(line_str) > 10
        return is_en and is_size

    @staticmethod
    def draw_clz_dict(clz_dict, img_bgr, out_path):
        color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for jdx, clz in enumerate(clz_dict.keys()):
            box_list = clz_dict[clz]
            if box_list:
                box_list, _ = filer_boxes_by_size(box_list)
                # print('[Info] box_list: {}'.format(box_list))
                img_bgr = draw_box_list(img_bgr, box_list, color=color_list[int(clz)], is_new=False)
                cv2.imwrite(out_path, img_bgr)

    @staticmethod
    def process_item(idx, data_line, idet, out_path):
        print('-' * 50)
        print('[Info] 开始: {}'.format(idx))
        data_dict = json.loads(data_line)
        # print('[Info] item: {}'.format(data_dict))
        image_url = data_dict["image_url"]
        angle = data_dict["angle"]
        ocr_result = data_dict["ocr_result"]
        # print('[Info] 手写行数: {}'.format(len(ocr_result)))
        _, img_bgr = download_url_img(image_url)
        img_bgr = rotate_img_for_4angle(img_bgr, angle)  # 旋转图像

        n_alter = 0
        for hw_idx, hw_data in enumerate(ocr_result):
            pos_data = hw_data["pos"]
            word = hw_data["word"]
            rec_box = Processor.parse_pos_2_rec(pos_data)
            box = rec2bbox(rec_box)
            is_en = Processor.filter_en_line(word)
            if not is_en:
                continue
            img_patch = get_cropped_patch(img_bgr, box)
            # print('[Info] is_en: {}, {}'.format(is_en, word))
            # show_img_bgr(img_patch)
            clz_dict = idet.detect_image(img_patch)
            if 2 not in clz_dict.keys():  # 只筛选删除的选项
                continue
            # image_name = image_url.split("/")[-1].split(".")[0]
            # out_image_path = os.path.join(out_dir, "{}_{}.jpg".format(image_name, hw_idx))
            # Processor.draw_clz_dict(clz_dict, img_patch, out_image_path)
            out_dict = {
                "image_url": image_url,
                "angle": angle,
                "ocr_result": hw_data,
                "clz_dict": clz_dict
            }
            out_line = json.dumps(out_dict)
            write_line(out_path, out_line)
            n_alter += 1
        print('[Info] idx: {}, n_alter: {},  写入完成: {}'.format(idx, n_alter, image_url))

    @staticmethod
    def process(lines_idx, data_lines, out_path):
        print('[Info] 分区: {}, 文件数量: {}'.format(lines_idx, len(data_lines)))
        idet = ImgDetector()
        # out_dir = os.path.join(DATA_DIR, 'en_full_dir')
        # mkdir_if_not_exist(out_dir)
        for idx, data_line in enumerate(data_lines):
            Processor.process_item(idx, data_line, idet, out_path)
            # break
        print('[Info] 分区: {}, 完成'.format(lines_idx))

    def process_mul(self):
        data_lines = read_file(self.data_path)
        n_prc = 4
        print('[Info] 进程数: {}'.format(n_prc))
        lines_param = []
        gap = len(data_lines) // n_prc
        for i in range(n_prc):
            lines_param.append(data_lines[i*gap:(i+1)*gap])
        pool = Pool(n_prc)
        for idx, lines in enumerate(lines_param):
            # Processor.process(idx, lines, self.out_path)
            pool.apply_async(Processor.process, (idx, lines, self.out_path))
            # break
        pool.close()
        pool.join()
        print('*' * 100)
        print('[Info] 处理完成: {}'.format(self.out_path))

    def data_checker(self):
        file_path = os.path.join(DATA_DIR, 'en_full_with_alter.out-20210715235808.txt')
        out_path = os.path.join(DATA_DIR, 'en_full_with_alter.out-filter.txt')
        create_file(out_path)
        out_dir = os.path.join(DATA_DIR, "en_full_with_alter")
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(file_path)
        # random.seed(47)
        # random.shuffle(data_lines)
        for idx, data_line in enumerate(data_lines):
            # if idx == 50:
            #     break
            data_dict = json.loads(data_line)
            image_url = data_dict["image_url"]
            angle = data_dict["angle"]
            pos_data = data_dict["ocr_result"]
            clz_dict = data_dict["clz_dict"]
            # _, img_bgr = download_url_img(image_url)
            # img_bgr = rotate_img_for_4angle(img_bgr, angle)  # 旋转图像
            rec_box = Processor.parse_pos_2_rec(pos_data)
            box = rec2bbox(rec_box)
            w = box[2] - box[0]
            h = box[3] - box[1]
            ratio = w // h
            area = w*h
            if ratio <= 7 or area < 10000:
                continue
            # print('[Info] area: {}, ratio: {}'.format(w*h, ratio))
            # img_patch = get_cropped_patch(img_bgr, box)
            # image_name = image_url.split("/")[-1].split(".")[0]
            # out_image_path = os.path.join(out_dir, "{}_{}.jpg".format(image_name, idx))
            # Processor.draw_clz_dict(clz_dict, img_patch, out_image_path)
            write_line(out_path, data_line)
            print('[Info] 处理完成: {}'.format(idx))
        print("[Info] 处理完成: {}".format(out_dir))


def main():
    pro = Processor()
    # pro.process_mul()
    pro.data_checker()


if __name__ == '__main__':
    main()
