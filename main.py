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

def get_indexes_lower_fps(
    num_frames: int,
    base_fps: int,
    new_fps: int,
) -> tuple:

    new_frames = int(num_frames * new_fps / base_fps)
    base_idx = np.arange(num_frames)
    extended_idx = np.repeat(base_idx, new_frames)
    mask = np.arange(len(extended_idx)) % num_frames == 0

    return extended_idx[mask]


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
        base_fps = capture.get(cv2.CAP_PROP_FPS)
        lower_fps = cfg['dataset']['frame_per_second']
        # cannot be more than base
        lower_fps = min(lower_fps, base_fps)
        video_idxs = get_indexes_lower_fps(
            num_frames=length,
            base_fps=base_fps,
            new_fps=lower_fps,
        )
    else:
        raise FileNotFoundError('Check imput file format.')

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
    counter_frame_id = 0
    read_idx = 0
    key_frame = False
    annotation_id = 0
    progress_bar = tqdm.tqdm(total=length)

    dir_with_video_path = os.path.dirname(video_path)

    if save_images:
        dir_with_images = os.path.join(dir_with_video_path, 'dataset')
        os.makedirs(dir_with_images, exist_ok=True)

    not_error = True

    while not_error:

        buffer_batched_frames = []

        for _ in range(cfg['model']['batch']):

            if not key_frame:
                if counter_frame_id < len(video_idxs):
                    read_idx_now = video_idxs[counter_frame_id]
                else:
                    not_error = False
                    break
            else:
                # TODO will raise an error with small number of frames
                read_idx_now = read_idx
            
            capture.set(cv2.CAP_PROP_POS_FRAMES, read_idx_now)
            not_error, original_frame = capture.read()

            if not not_error:
                break

            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            buffer_batched_frames.append(frame)
            
            if not key_frame:
                read_idx = video_idxs[counter_frame_id] + 1
                counter_frame_id += 1
            else:
                read_idx += 1
                if read_idx > video_idxs[counter_frame_id]:
                    counter_frame_id += 1

        if not not_error:
            break

        tick = time.perf_counter()
        predicted_instances = inference_detector(model, buffer_batched_frames)
        tack = time.perf_counter()

        # for each image
        for i in range(cfg['model']['batch']):

            bboxs, masks = select_instances(
                predicted_instances[i],
                cfg['model']['target_classes'],
                cfg['model']['confidence_treshold'],
            )

            image_height, image_width, _ = buffer_batched_frames[i].shape
            output_image_path = os.path.abspath(os.path.join(dir_with_images, f'{counter}.jpg'))

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
            if key_frame:
                delta = 1
            else:
                id_prev = counter_frame_id - cfg['model']['batch'] + i
                id_prev_prev = counter_frame_id - cfg['model']['batch'] + i - 1
                if id_prev_prev < 0:
                    id_prev_prev = 0
                delta = video_idxs[id_prev] - video_idxs[id_prev_prev]
            progress_bar.update(delta)

        if num_objects > 0:
            key_frame = True
        else:
            key_frame = False

    with open(os.path.join(dir_with_video_path, 'train.json'), 'w') as f:
        json.dump(output_data, f)

if __name__ == '__main__':
    main()
