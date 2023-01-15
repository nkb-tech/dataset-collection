# dataset-collection
Framework to collect dataset in COCO format for images/videos using pretrained neural networks.

# Table of contents
1. [Introduction](#introduction)
2. [Datasets Description](#paragraph1)
    1. [Youtube VIS](#subparagraph1)
    2. [MSCOCO](#subparagraph2)
3. [How to run](#paragraph2)
3. [Contribute](#paragraph3)

## Introduction <a name="introduction"></a>
Framework to collect dataset in COCO format for images/videos using pretrained neural networks.

## Datasets Description <a name="paragraph1"></a>
There are supported datasets.

### Youtube VIS <a name="subparagraph1"></a>
<details open>
<summary href="https://youtube-vos.org/dataset/vis/">Annotation example</summary>
{
    "videos" : [video],
    "annotations" : [annotation],
    "categories" : [category],
}
video{
    "id" : int,
    "width" : int,
    "height" : int,
    "length" : int,
    "file_names" : [file_name],
}
annotation{
    "id" : int,
    "video_id" : int,
    "category_id" : int,
    "segmentations" : [RLE or [polygon] or None],
    "areas" : [float or None],
    "bboxes" : [[x,y,width,height] or None],
    "iscrowd" : 0 or 1,
}
category{
    "id" : int,
    "name" : str,
    "supercategory" : str,
}
</details>

### MSCOCO <a name="subparagraph2"></a>
<details open>
<summary href="https://cocodataset.org/#format-data/">Annotation example</summary>
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category],
}
image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}
annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
category{
    "id": int,
    "name": str,
    "supercategory": str,
}
</details>

## How to run <a name="paragraph2"></a>

```bash
git clone https://github.com/msb-tech/dataset-collection.git
cd dataset-collection
```

Before running you should change the default values in `docker_names.sh`. After that type

```bash
source docker/docker_names.sh
bash docker/build+run.sh
python3 tools/process_data.py -c config.py -s
```

To see help info about the script, type

```bash
python3 tools/process_data.py --help
```

## Contribute <a name="paragraph3"></a>

```bash
pip install pre-commit
pre-commit install
```
