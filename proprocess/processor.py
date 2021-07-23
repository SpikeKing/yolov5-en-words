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
        print('[Info] 输入文件: {}'.format(self.data_path))
        print('[Info] 输出文件: {}'.format(self.out_path))
        # 已处理的图像
        self.used1_path = os.path.join(DATA_DIR, 'en_full_with_alter.anno-v1_1.txt')
        self.used2_path = os.path.join(DATA_DIR, 'en_lowscore.anno-v1_2.txt')
        print('[Info] 已处理文件1: {}'.format(self.used1_path))
        print('[Info] 已处理文件: {}'.format2(self.used2_path))

    def get_used_urls(self):
        """
        获取已使用的urls
        """
        data1_lines = read_file(self.used1_path)
        data2_lines = read_file(self.used2_path)
        data_lines = data1_lines + data2_lines
        url_boxes_dict = collections.defaultdict(list)
        for data_line in data_lines:
            data_dict = json.loads(data_line)
            image_url = data_dict["image_original_url"]
            anno_data = data_dict["polygon_annotation"]
            for anno in anno_data:
                anno_type = anno["type"]
                if anno_type == "WenBenHang":
                    url_boxes_dict[image_url].append(tuple(anno["coords"]))
        print('[Info] url_boxes_dict: {}'.format(len(url_boxes_dict.keys())))
        return url_boxes_dict

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
                # box_list, _ = filer_boxes_by_size(box_list)
                # img_bgr = draw_box_list(img_bgr, box_list, color=color_list[int(clz)], is_new=False)
                img_bgr = draw_rec_list(img_bgr, box_list, color=color_list[int(clz)], is_new=False)
                cv2.imwrite(out_path, img_bgr)

    @staticmethod
    def save_img_path(img_bgr, img_name):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        oss_root_dir = "zhengsheng.wcl/Character-Detection/datasets/english-words-imgs/"
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_item(idx, data_line, idet, url_boxes_dict, out_path):
        print('-' * 50)
        print('[Info] 开始: {}'.format(idx))
        data_dict = json.loads(data_line)
        # print('[Info] item: {}'.format(data_dict))
        image_url = data_dict["image_url"]
        angle = data_dict["angle"]
        if angle != 0:  # 过滤非0图像
            return
        ocr_result = data_dict["ocr_result"]
        # print('[Info] 手写行数: {}'.format(len(ocr_result)))
        _, img_bgr = download_url_img(image_url)
        # img_bgr = rotate_img_for_4angle(img_bgr, angle)  # 旋转图像

        used_boxes = url_boxes_dict[image_url]
        n_alter = 0
        for hw_idx, hw_data in enumerate(ocr_result):
            pos_data = hw_data["pos"]
            word = hw_data["word"]
            rec_classify = hw_data["recClassify"]  # 手写25
            if rec_classify != 25:  # 25是手写标记
                continue
            rec_box = Processor.parse_pos_2_rec(pos_data)
            box = rec2bbox(rec_box)
            if tuple(bbox2rec(box)) in used_boxes:
                print('[Info] 已使用: {}'.format(box))
                continue
            is_en = Processor.filter_en_line(word)
            if not is_en:
                continue
            img_patch = get_cropped_patch(img_bgr, box)
            # print('[Info] is_en: {}, {}'.format(is_en, word))
            clz_dict = idet.detect_image(img_patch)
            if "2" not in clz_dict.keys():  # 只筛选删除的选项
                continue

            # show_img_bgr(img_patch)
            image_name_x = image_url.split("/")[-1].split(".")[0]
            image_name = "{}.jpg".format(image_name_x)
            image_convert_url = Processor.save_img_path(img_bgr, image_name)
            # image_name = image_url.split("/")[-1].split(".")[0]
            # out_image_path = os.path.join(out_dir, "{}_{}.jpg".format(image_name, hw_idx))
            # Processor.draw_clz_dict(clz_dict, img_patch, out_image_path)
            out_dict = {
                "image_url": image_convert_url,
                "image_original_url": image_url,
                "angle": angle,
                "hw_data": hw_data,
                "clz_dict": clz_dict
            }
            out_line = json.dumps(out_dict)
            write_line(out_path, out_line)
            n_alter += 1
        print('[Info] idx: {}, 涂改: {},  完成: {}'.format(idx, n_alter, image_url))

    @staticmethod
    def process(lines_idx, data_lines, url_boxes_dict, out_path):
        print('[Info] 分区: {}, 文件数量: {}'.format(lines_idx, len(data_lines)))
        idet = ImgDetector()
        # out_dir = os.path.join(DATA_DIR, 'en_full_dir')
        # mkdir_if_not_exist(out_dir)
        for idx, data_line in enumerate(data_lines):
            Processor.process_item(idx, data_line, idet, url_boxes_dict, out_path)
            # break
        print('[Info] 分区: {}, 完成'.format(lines_idx))

    def process_mul(self):
        url_boxes_dict = self.get_used_urls()
        data_lines = read_file(self.data_path)
        n_prc = 4
        print('[Info] 进程数: {}'.format(n_prc))
        lines_param = []
        gap = len(data_lines) // n_prc
        for i in range(n_prc):
            lines_param.append(data_lines[i*gap:(i+1)*gap])
        pool = Pool(n_prc)
        for idx, lines in enumerate(lines_param):
            # Processor.process(idx, lines, url_boxes_dict, self.out_path)
            pool.apply_async(Processor.process, (idx, lines, url_boxes_dict, self.out_path))
            break
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

    @staticmethod
    def format_word_dict(clz_dict, hw_rec_box):
        hw_bbox = rec2bbox(hw_rec_box)
        x_min, y_min = hw_bbox[0], hw_bbox[1]
        res_dict = collections.defaultdict(list)
        type_dict = {"0": "YingYuDanCi", "1": "ZhongWenShouXie", "2": "TuGai"}
        item_list = []
        for clz in clz_dict.keys():
            box_list = clz_dict[clz]
            for box in box_list:
                box = [box[0] + x_min, box[1] + y_min, box[2] + x_min, box[3] + y_min]
                rec = bbox2rec(box)
                type_name = type_dict[clz]
                item = {"coords": rec, "type": type_name}
                item_list.append(item)
        return item_list

    @staticmethod
    def process_img_line(image_data_list, image_url, out_path, img_idx):
        angle, _, _ = image_data_list[0]
        _, img_bgr = download_url_img(image_url)
        img_bgr = rotate_img_for_4angle(img_bgr, angle)  # 旋转图像
        height, width, _ = img_bgr.shape

        image_name_x = image_url.split("/")[-1].split(".")[0]
        image_name = "{}_{}.jpg".format(image_name_x, angle)
        image_url = Processor.save_img_path(img_bgr, image_name)

        res_items = []
        for image_data in image_data_list:
            _, pos_data, clz_dict = image_data
            hw_rec_box = Processor.parse_pos_2_rec(pos_data)
            # print('[Info] rec_box: {}'.format(hw_rec_box))
            item = {"coords": hw_rec_box, "type": "WenBenHang"}
            item_list = Processor.format_word_dict(clz_dict, hw_rec_box)
            # print('[Info] clz_dict: {}'.format(clz_dict))
            res_items.append(item)
            res_items += item_list

        res_dict = {
            "image_url": image_url,
            "height": str(height),
            "width": str(width),
            "polygon_annotation": res_items
        }

        res_dict_str = json.dumps(res_dict)
        write_line(out_path, res_dict_str)
        print('[Info] {} 处理完成: {}'.format(img_idx, image_url))

    def format_data(self):
        img_path = os.path.join(DATA_DIR, 'en_full_with_alter.out-filter.txt')
        out_path = os.path.join(DATA_DIR, 'en_full_with_alter.anno-{}.txt'.format(get_current_time_str()))
        data_lines = read_file(img_path)
        print('[Info] 文本行数: {}'.format(len(data_lines)))
        image_dict = collections.defaultdict(list)
        for data_line in data_lines:
            data_dict = json.loads(data_line)
            image_url = data_dict["image_url"]
            angle = data_dict["angle"]
            pos_data = data_dict["ocr_result"]
            clz_dict = data_dict["clz_dict"]
            image_dict[image_url].append([angle, pos_data, clz_dict])
        print('[Info] 图像数: {}'.format(len(image_dict.keys())))

        pool = Pool(10)
        for img_idx, image_url in enumerate(image_dict.keys()):
            image_data_list = image_dict[image_url]
            # Processor.process_img_line(image_data_list, image_url, out_path, img_idx)
            pool.apply_async(Processor.process_img_line, (image_data_list, image_url, out_path, img_idx))

        pool.close()
        pool.join()
        print('[Info] 全部完成: {}'.format(out_path))

    def anno_check(self):
        img_path = os.path.join(DATA_DIR, 'en_full_with_alter.anno-20210716155344.txt')
        out_dir = os.path.join(DATA_DIR, 'en_full_with_alter_anno')
        mkdir_if_not_exist(out_dir)
        data_lines = read_file(img_path)
        random.seed(47)
        random.shuffle(data_lines)
        print('[Info] 数据行数: {}'.format(len(data_lines)))
        for idx, data_line in enumerate(data_lines):
            if idx == 20:
                break
            data_dict = json.loads(data_line)
            image_url = data_dict["image_url"]
            out_name = image_url.split("/")[-1].split(".")[0]
            out_name = "{}-check.jpg".format(out_name)
            out_path = os.path.join(out_dir, out_name)
            polygon_annotation = data_dict["polygon_annotation"]
            rec_list = []
            for pa in polygon_annotation:
                coords = pa["coords"]
                rec_list.append(coords)
            _, img_bgr = download_url_img(image_url)
            draw_rec_list(img_bgr, rec_list, save_name=out_path)


def main():
    pro = Processor()
    pro.process_mul()
    # pro.data_checker()
    # pro.format_data()
    # pro.anno_check()


if __name__ == '__main__':
    main()
