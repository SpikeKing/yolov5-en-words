#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 19.7.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from detect_image import ImgDetector
from myutils.cv_utils import *
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
    def draw_clz_dict(clz_dict, img_bgr, out_path):
        color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for jdx, clz in enumerate(clz_dict.keys()):
            box_list = clz_dict[clz]
            if box_list:
                img_bgr = draw_box_list(img_bgr, box_list, color=color_list[int(clz)], is_new=False)
                cv2.imwrite(out_path, img_bgr)

    @staticmethod
    def process_item(idx, data_line, detector, out_path):
        print('[Info] 处理开始: {}'.format(idx))

        image_url = data_line[0]
        ocr_result = data_line[1]

        _, img_bgr = download_url_img(image_url)
        height, width, _ = img_bgr.shape

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
            "image_url": image_url,
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
            ProcessorV2.process_item(idx, data_line, detector, out_path)
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

    def check_data(self):
        data_path = os.path.join(DATA_DIR, 'en_lowscore.anno-20210720114508.txt')
        out_dir = os.path.join(DATA_DIR, 'en_lowscore_anno')
        mkdir_if_not_exist(out_dir)
        data_lines = read_file(data_path)
        random.seed(47)
        random.shuffle(data_lines)
        print('[Info] 数据行数: {}'.format(len(data_lines)))
        for idx, data_line in enumerate(data_lines):
            if idx == 20:
                break
            data_dict = json.loads(data_line)
            image_url = data_dict["image_url"]
            _, img_bgr = download_url_img(image_url)
            polygon_annotation = data_dict["polygon_annotation"]

            out_name = image_url.split("/")[-1]
            rec_list = []
            for p_idx, pa in enumerate(polygon_annotation):
                print(pa)
                rec = pa["coords"]
                rec_list.append(rec2bbox(rec))
                draw_box(img_bgr, rec2bbox(rec), is_show=True)
            draw_box_list(img_bgr, rec_list, is_show=False,
                          is_text=False, save_name=os.path.join(out_dir, out_name))


def main():
    prc2 = ProcessorV2()
    prc2.process()
    # prc2.check_data()


if __name__ == '__main__':
    main()
