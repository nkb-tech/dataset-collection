import click
import tqdm
import os
import time
import cv2
import mmcv
import mmdet
import json
import yaml
import numpy as np
import pycocotools.mask as mask_utils
from mmdet.apis import inference_detector, init_detector


idx_category = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}
category_mapping = {idx: category for idx, category in enumerate(idx_category.values())}
idx_mapping = {category: idx for idx, category in category_mapping.items()}

def read_from_file(
    cfg_file: str,
) -> dict:

    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

    return new_config

def select_instances(
    predicted_instances: tuple,
    target_classes: list=['dog', ],
    treshold_confidences: list=[1e-2, ],
    idx_mappping: dict=idx_mapping,
) -> tuple:

    """
    Selects target classes above treshold confidence from output mmdetection model
    """

    bboxes, masks = predicted_instances['ins_results']

    target_idxs = [idx_mappping[target_class] for target_class in target_classes]

    target_bboxes, target_masks = [], []

    for target_idx, target_treshold_confidence in zip(target_idxs, treshold_confidences):
        filtered_bboxes, filtered_masks = [], []
        for bbox, mask in zip(bboxes[target_idx], masks[target_idx]):
            if len(bbox) == 0:
                continue
            x_l, y_l, x_r, y_r, sc = bbox
            if sc > target_treshold_confidence:
                filtered_bboxes.append(bbox)
                filtered_masks.append(mask)
        target_bboxes.append(np.array(filtered_bboxes))
        target_masks.append(np.array(filtered_masks))

    target_result = (target_bboxes, target_masks)

    return target_result


@click.command()
@click.option(
    '--config_path', '-c',
    default='config.yaml',
    type=str,
    help='Input config path.',
)
@click.option(
    '--save_images', '-s',
    default=False,
    type=bool,
    is_flag=True,
    help='Save images from video or not',
)
def main(
    config_path: str,
    save_images: bool,
) -> None:

    # read a config
    cfg = read_from_file(config_path)

    # build the model
    model = init_detector(
        cfg['model']['mmconfig'],
        cfg['model']['mmcheckpoint'],
        cfg['model']['device'],
    )

    # Convert the model into evaluation mode
    model.eval()

    video_path = cfg['dataset']['input_path']

    if video_path[video_path.rfind('.') + 1:].lower() in cfg['dataset']['file_extensions']:
        capture = cv2.VideoCapture(video_path)
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        print('Waiting for video file')

    output_data = {
        'images': [],
        'annotations': [],
        'categories': [],
    }

    for i, target_class in enumerate(cfg['model']['target_classes']):
        category = {
            'id': i + 1,
            'name': target_class,
            'supercategory': target_class,
        }
        output_data['categories'].append(category)

    counter = 0
    annotation_id = 0
    progress_bar = tqdm.tqdm(total=length)

    dir_with_video_path = os.path.dirname(video_path)

    if save_images:
        dir_with_images = os.path.join(dir_with_video_path, 'dataset')
        os.makedirs(dir_with_images, exist_ok=True)

    while True:

        ret, original_frame = capture.read()

        output_image_path = os.path.abspath(os.path.join(dir_with_images, f'{counter}.jpg'))

        if not ret:
            break

        frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape

        tick = time.perf_counter()
        predicted_instances = inference_detector(model, frame)
        tack = time.perf_counter()

        bboxs, masks = select_instances(
            predicted_instances,
            cfg['model']['target_classes'],
            cfg['model']['confidence_treshold'],
        )

        image_info = {
            'id': counter,
            'width': image_width,
            'height': image_height,
            'file_name': output_image_path,
        }

        num_objects = 0

        # for each class
        for j in range(len(bboxs)):
            # in class for each object
            for bbox, mask in zip(bboxs[j], masks[j]):

                x_l, y_l, x_r, y_r, _ = bbox

                x = (x_r + x_l) / 2
                y = (y_r + y_l) / 2
                width = x_r - x_l
                height = y_r - y_l

                rle_encoded = mask_utils.encode(np.asfortranarray(mask.astype(bool)))

                annotation = {
                    'id': annotation_id,
                    'image_id': counter,
                    'class_id': j + 1,
                    'area': int(width * height),
                    'bbox': [int(x), int(y), int(width), int(height)],
                    'segmentation': {
                        'size': rle_encoded['size'],
                        'counts': rle_encoded['counts'].decode(encoding='UTF-8'),
                    },
                    'iscrowd': 0,
                }
                annotation_id += 1

                output_data['annotations'].append(annotation)

                num_objects += 1 

        if num_objects > 0:
            output_data['images'].append(image_info)
            if save_images:
                cv2.imwrite(output_image_path, original_frame)

        counter += 1

        progress_bar.set_postfix({
            'num_objects': num_objects,
            'detect_time (FPS)': 1 / (tack - tick),
        })
        progress_bar.update()

    with open(os.path.join(dir_with_video_path, 'train.json'), 'w') as f:
        json.dump(output_data, f)

if __name__ == '__main__':
    main()
