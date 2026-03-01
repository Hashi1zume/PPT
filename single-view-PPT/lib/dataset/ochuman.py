# ------------------------------------------------------------------------------
# OCHuman Dataset (COCO format)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import defaultdict
from collections import OrderedDict

from nms.nms import oks_nms
from nms.nms import soft_oks_nms
from dataset.coco import COCODataset

logger = logging.getLogger(__name__)


class OCHumanDataset(COCODataset):
    """OCHuman (COCO形式) 用のDataset.

    image_setに指定されたファイル名をそのまま参照できるようにする。
    例: image_set = "ochuman_coco_format_test_range_0.00_1.00.json"
    """

    def _get_ann_file_keypoint(self):
        if self.image_set.endswith('.json'):
            ann_file = self.image_set
        else:
            ann_file = self.image_set + '.json'
        return os.path.join(self.root, 'annotations', ann_file)

    def image_path_from_index(self, index):
        """OCHumanは images/ 直下に格納されるためCOCOのsubdirを使わない"""
        img_info = self.coco.loadImgs(index)[0]
        file_name = img_info['file_name']
        return os.path.join(self.root, 'images', file_name)

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        """OCHumanはCOCOのファイル名規則に依存しないためID解決を上書きする"""
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # file_name -> image_id の対応表
        file_to_id = {v['file_name']: k for k, v in self.coco.imgs.items()}

        _kpts = []
        for idx, kpt in enumerate(preds):
            file_name = os.path.basename(img_path[idx])
            image_id = file_to_id.get(file_name, None)
            if image_id is None:
                raise ValueError('Unknown image file in annotations: {}'.format(file_name))

            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': image_id
            })

        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        info_str = self._do_python_keypoint_eval(res_file, res_folder)
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

