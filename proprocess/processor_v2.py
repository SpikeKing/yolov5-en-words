#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 19.7.21
"""

import os
import sys
from multiprocessing.pool import Pool

import cv2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from detect_image import ImgDetector
from myutils.cv_utils import bbox2rec, rec2bbox, get_cropped_patch, draw_rec_list, show_img_bgr
from myutils.project_utils import *
from root_dir import DATA_DIR


class ProcessorV2(object):
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, 'en_lowscore.json')
        time_str = get_current_time_str()
        self.out_path = os.path.join(DATA_DIR, 'en_lowscore.anno-{}.txt'.format(time_str))

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
    def filter_box_size(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        ratio = w // h
        area = w * h
        if ratio <= 7 or area < 10000:
            return False
        else:
            return True

    @staticmethod
    def format_word_dict(clz_dict, hw_box):
        x_min, y_min = hw_box[0], hw_box[1]
        type_dict = {"0": "YingYuDanCi", "1": "ZhongWenShouXie", "2": "TuGai"}
        item_list = []
        for clz in clz_dict.keys():
            box_list = clz_dict[clz]
            for box in box_list:
                box = [box[0] + x_min, box[1] + y_min, box[2] + x_min, box[3] + y_min]
                rec = bbox2rec(box)
                type_name = type_dict[str(clz)]
                item = {"coords": rec, "type": type_name}
                item_list.append(item)
        return item_list

    @staticmethod
    def save_img_patch(img_bgr, img_name):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        oss_root_dir = "zhengsheng.wcl/Character-Detection/datasets/english-words-imgs-{}/"\
            .format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_item(idx, data_line, detector, out_path):
        print('[Info] 处理开始: {}'.format(idx))

        image_url = data_line[0]
        ocr_result = data_line[1]

        _, img_bgr = download_url_img(image_url)
        height, width, _ = img_bgr.shape
        image_name = image_url.split("/")[-1]
        image_url_convert = ProcessorV2.save_img_patch(img_bgr, image_name)

        n_alter = 0
        res_items = []
        for hw_idx, hw_data in enumerate(ocr_result):
            pos_data = hw_data["pos"]
            word = hw_data["word"]
            rec_box = ProcessorV2.parse_pos_2_rec(pos_data)
            hw_box = rec2bbox(rec_box)

            # 过滤
            # is_en = ProcessorV2.filter_en_line(word)
            # is_size = ProcessorV2.filter_box_size(hw_box)
            # if not is_en or not is_size:
            #     continue

            img_patch = get_cropped_patch(img_bgr, hw_box)
            clz_dict = detector.detect_image(img_patch)
            item_list = ProcessorV2.format_word_dict(clz_dict, hw_box)
            hw_item = {"coords": bbox2rec(hw_box), "type": "WenBenHang"}
            res_items += (item_list + [hw_item])

        res_dict = {
            "image_url": image_url_convert,
            "image_original_url": image_url,
            "height": str(height),
            "width": str(width),
            "polygon_annotation": res_items
        }
        res_dict_str = json.dumps(res_dict)
        write_line(out_path, res_dict_str)

        print('[Info] idx: {}, n_alter: {}, 处理完成: {}'.format(idx, n_alter, image_url))

    @staticmethod
    def process_block(block_idx, data_lines, out_path):
        print('[Info] 分区: {}, 文件数量: {}'.format(block_idx, len(data_lines)))
        detector = ImgDetector()
        for idx, data_line in enumerate(data_lines):
            try:
                ProcessorV2.process_item(idx, data_line, detector, out_path)
            except Exception as e:
                print(e)
        print('[Info] 分区: {}, 完成'.format(block_idx))

    def process(self):
        """
        处理
        """
        data_raw = read_file(self.data_path)[0]
        data_dict = json.loads(data_raw)
        data_lines = [(k, v) for k, v in data_dict.items()]

        n_prc = 4
        print('[Info] 进程数: {}'.format(n_prc))
        lines_param = []
        gap = len(data_lines) // n_prc
        for i in range(n_prc):
            lines_param.append(data_lines[i * gap:(i + 1) * gap])
        pool = Pool(n_prc)
        for idx, lines in enumerate(lines_param):
            # ProcessorV2.process_block(idx, lines, self.out_path)
            pool.apply_async(ProcessorV2.process_block, (idx, lines, self.out_path))
        pool.close()
        pool.join()
        print('*' * 100)
        print('[Info] 处理完成: {}'.format(self.out_path))

    def modify_axis(self, boxes, base_box):
        x_min, y_min, _, _ = base_box
        res_boxes = []
        for box in boxes:
            box = box["coords"]
            box = rec2bbox(box)
            box = [box[0] - x_min, box[1] - y_min, box[2] - x_min, box[3] - y_min]
            res_boxes.append(box)
        return res_boxes

    def draw_polygon_annotation(self, img_bgr, polygon_annotation, save_name):
        word_rec_list = []
        text_rec_list = []
        for p_idx, pa in enumerate(polygon_annotation):
            rec = pa["coords"]
            type_str = pa["type"]
            if type_str != "WenBenHang":
                word_rec_list.append(rec)
            else:
                text_rec_list.append(rec)
        img_bgr = draw_rec_list(img_bgr, word_rec_list, is_show=False, thickness=-1)
        show_img_bgr(img_bgr)
        draw_rec_list(img_bgr, text_rec_list, is_show=False, thickness=5, save_name=save_name)
        show_img_bgr(img_bgr)

    def check_data(self):
        data_path = os.path.join(DATA_DIR, 'en_lowscore.anno-v1_2.txt')
        out_dir = os.path.join(DATA_DIR, 'en_lowscore_anno')
        mkdir_if_not_exist(out_dir)
        data_lines = read_file(data_path)
        random.seed(47)
        random.shuffle(data_lines)
        print('[Info] 数据行数: {}'.format(len(data_lines)))
        for idx, data_line in enumerate(data_lines):
            # if idx == 5:
            #     break
            data_dict = json.loads(data_line)
            image_url = data_dict["image_url"]
            out_name = image_url.split("/")[-1]

            if out_name != "0900c7acdf107184d65956a785e55672.jpg":
                continue

            _, img_bgr = download_url_img(image_url)
            cv2.imwrite(os.path.join(DATA_DIR, 'tmp.jpg'), img_bgr)
            img_bgr = cv2.imread(os.path.join(DATA_DIR, 'tmp.jpg'))
            polygon_annotation = data_dict["polygon_annotation"]

            out_img_path = os.path.join(out_dir, out_name)
            self.draw_polygon_annotation(img_bgr, polygon_annotation, out_img_path)
            print('[Info] 绘制: {}'.format(out_img_path))


def main():
    prc2 = ProcessorV2()
    # prc2.process()
    prc2.check_data()


if __name__ == '__main__':
    main()
