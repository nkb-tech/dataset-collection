seed = 0
launcher = None
deterministic = True

model = dict(
    task='segm',
    mmconfig='/home/user/mmlab/mmdet/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py',
    mmcheckpoint='https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth',
    target_classes=['dog', ],
    confidence_threshold=[0.1, ],
    device='cuda:0',
    batch=1,
)

dataset = dict(
    dataset_type='coco',
    input_path='/home/user/src/project/data/14.08/19.00_до_22.00/PVN_hd_ZAO_10297_1.mp4',
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
