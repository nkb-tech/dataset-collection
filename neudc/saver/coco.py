import json
import logging
import os
import os.path as osp
import typing as tp

import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_utils
import torch
import yaml
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.utils import get_root_logger


class BaseSaver(object):
    def __init__(self,
                 output_path: str,
                 all_classes_config: str,
                 target_classes: tp.List[str],
                 threshold_confidences: tp.List[float],
                 save_images: bool = True,
                 debug: bool = True,
                 save_as: str = 'jpg',
                 log_file: str = None) -> None:
        '''
        Saver for COCO dataset.
        Args:
            output_path (str): where to save the data
            all_classes_config (str): path to default classes
                config
            save_images (bool): save pure images or not
            debug (bool): save or not prediction results
            target_classes (list[str]): names of target
                classes.
            threshold_confidences (list[str]):
                confidences for each class or one for all
            save_as (str): save data format
            log_file (str): file path for saving logs
        '''

        self.output_path = output_path
        self.debug = debug
        self.save_images = save_images
        self.save_as = save_as

        if self.save_images:
            self.images_path = osp.join(output_path, 'images')
            os.makedirs(self.images_path, exist_ok=True)

        if self.debug:
            self.debug_path = osp.join(output_path, 'debug')
            os.makedirs(self.debug_path, exist_ok=True)

        if isinstance(threshold_confidences, float):
            threshold_confidences = [threshold_confidences] * len(
                target_classes)  # noqa

        assert len(threshold_confidences) == len(target_classes), (
            f'Lengths should be the same, got for confs'
            f'{len(threshold_confidences)} and for classes'
            f'{len(target_classes)}.')

        self.threshold_confidences = threshold_confidences
        self.target_classes = target_classes

        with open(all_classes_config, 'r') as file:
            all_classes = yaml.safe_load(file)

        # TODO maybe it is not needed :)
        # class mapping
        # at first, align idxs
        category_mapping = {
            idx: category
            for idx, category in enumerate(all_classes.values())
        }
        # at second, reverse dict
        idx_mapping = {
            category: idx
            for idx, category in category_mapping.items()
        }
        # get indexes
        self.target_idxs = [
            idx_mapping[target_class] for target_class in self.target_classes
        ]

        # logging
        self.logger = get_root_logger(
            log_file=log_file,
            log_level=logging.INFO,
        )

    def clear_database(self) -> None:
        pass

    def __call__(self, *args, **kwargs) -> tp.List[bool]:

        pass


class COCOSaver(BaseSaver):
    def __init__(
            self,
            output_path: str,
            target_classes: tp.List[str],
            threshold_confidences: tp.List[float],
            all_classes_config:
        str = 'configs/datasets/coco/id2class.yaml',  # noqa
            save_images: bool = True,
            debug: bool = True,
            with_mask: bool = True,
            log_file: str = None) -> None:
        '''
        Saver for COCO dataset.
        Args:
            output_path (str): where to save the data
            all_classes_config (dict[int, str]): trained config
                for classes
            save_images (bool): save pure images or not
            debug (bool): save or not prediction results
            with_mask (bool): using instance mask or not
            target_classes (list[str]): names of target classes
            threshold_confidences (list[str]):
                confidences for each class or one for all
            log_file (str): file path for saving logs
        '''

        super(COCOSaver,
              self).__init__(output_path=output_path,
                             all_classes_config=all_classes_config,
                             save_images=save_images,
                             debug=debug,
                             target_classes=target_classes,
                             threshold_confidences=threshold_confidences,
                             log_file=log_file)

        assert with_mask, 'Now only with_mask=True supported.'

        # TODO check for different models if preds have this field
        self.mmdet_field = 'ins_results' if with_mask else 'results'

        self.clear_database()

    def clear_database(self) -> None:

        # coco-based indexers
        self.annotation_id = 0
        self.image_id = 0

        # main coco database
        self.database = {
            'images': [],
            'annotations': [],
            'categories': [],
        }

        # fill out coco `categories`
        for i, target_class in enumerate(self.target_classes):
            category = {
                'id': i + 1,
                'name': target_class,
                'supercategory': target_class,
            }
            self.database['categories'].append(category)

    def __call__(self, batch_frames: tp.List[torch.Tensor],
                 batch_preds: tp.List[tp.Dict[str, torch.Tensor]],
                 batch_idxs: int) -> tp.List[bool]:

        batch_filter_mask = []

        # iterate over frames
        for frame_idx, preds, frame in zip(batch_idxs, batch_preds,
                                           batch_frames):

            # TODO `self.mmdet_field` can be not right
            # also output will depend on self.mmdet_field
            # tuple with different length
            if isinstance(preds, tuple):
                bboxes, masks = preds
            elif isinstance(preds, dict):
                bboxes, masks = preds[self.mmdet_field]
            else:
                raise NotImplementedError

            target_objects_on_frame = False

            image_height, image_width, _ = frame.shape
            output_image_path = f'{frame_idx}.{self.save_as}'

            image_info = {
                'id': int(frame_idx),
                'width': image_width,
                'height': image_height,
                'file_name': output_image_path,
            }

            # TODO for future just change mmdet function
            # to less coding. Vars only needed for visualisation
            target_bboxes, target_masks = [], []

            # iterate over all target classes
            for j, (target_idx, target_treshold_confidence) in enumerate(
                    zip(self.target_idxs, self.threshold_confidences)):

                filtered_bboxes, filtered_masks = [], []

                for bbox, mask in zip(bboxes[target_idx], masks[target_idx]):

                    if len(bbox) == 0:
                        continue

                    x_l, y_l, x_r, y_r, sc = bbox
                    if sc < target_treshold_confidence:
                        continue

                    filtered_bboxes.append(bbox)
                    filtered_masks.append(mask)
                    target_objects_on_frame = True

                    x = (x_r + x_l) / 2
                    y = (y_r + y_l) / 2
                    bbox_width = x_r - x_l
                    bbox_height = y_r - y_l

                    rle_encoded = mask_utils.encode(
                        np.asfortranarray(mask.astype(bool)))

                    mask_area = mask_utils.area(rle_encoded)

                    annotation = {
                        'id': self.annotation_id,
                        'image_id': int(frame_idx),
                        'class_id': j + 1,
                        'iscrowd': 0,
                        'area': int(mask_area),
                        'bbox':
                        [int(x),
                         int(y),
                         int(bbox_width),
                         int(bbox_height)],
                        'segmentation': {
                            'size':
                            rle_encoded['size'],
                            'counts':
                            rle_encoded['counts'].decode(encoding='UTF-8'),
                        },
                    }
                    self.annotation_id += 1
                    self.database['annotations'].append(annotation)

                target_bboxes.append(np.array(filtered_bboxes))
                target_masks.append(np.array(filtered_masks))

            if target_objects_on_frame:

                self.database['images'].append(image_info)

                if self.save_images:
                    filename = osp.join(self.images_path, output_image_path)
                    result = cv2.imwrite(
                        filename=filename,
                        img=frame,
                    )
                    if not result:
                        self.logger.error(
                            f'Frame {frame_idx} was not saved to {filename}')

                if self.debug:
                    bboxes = np.vstack(target_bboxes)
                    labels = np.concatenate([
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(target_bboxes)
                    ])

                    segms = mmcv.concat_list(target_masks)
                    if isinstance(segms[0], torch.Tensor):
                        segms = torch.stack(segms,
                                            dim=0).detach().cpu().numpy()
                    else:
                        segms = np.stack(segms, axis=0)

                    filename = osp.join(self.debug_path, output_image_path)

                    imshow_det_bboxes(
                        img=frame,
                        bboxes=bboxes,
                        labels=labels,
                        segms=segms,
                        class_names=self.target_classes,
                        show=False,
                        out_file=filename,
                    )

            batch_filter_mask.append(target_objects_on_frame)

        return batch_filter_mask

    def dump(self, filename: str = 'database.json') -> None:

        root, ext = osp.splitext(filename)

        if ext != '.json':
            filename = f'{root}.json'

        database_path = osp.join(self.output_path, filename)

        with open(database_path, 'w') as file:
            json.dump(self.database, file)

        self.logger.info(f'COCO-like database was written to {database_path}')
