seed = 0
launcher = None
deterministic = True

model = dict(
    task='segm',
    mmconfig=
    '/home/user/mmlab/mmdet/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py',  # noqa
    mmcheckpoint=
    'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth',  # noqa
    target_classes=[
        'dog',
    ],
    confidence_threshold=[
        0.1,
    ],
    device='cuda:0',
    batch_size=1,
)

dataset = dict(
    dataset_type='coco',
    num_threads=8,
    input_path='assets/example.mp4',
    recursive=False,
    output_path='data/custom_dataset',
    frame_per_second=1,
    high_fps_interval=2,
    file_extensions=[
        'mp4',
        'mov',
        'avi',
        'mkv',
    ],
)
