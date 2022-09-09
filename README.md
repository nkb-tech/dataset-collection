# dataset-collection
Framework to collect dataset in COCO format for images/videos using pretrained neural networks

# Datasets Description

<details open>
<summary href="https://youtube-vos.org/dataset/vis/">Youtube VIS</summary>
```
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
```
</details>

<details open>
<summary href="https://cocodataset.org/#format-data/">MSCOCO</summary>
```
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
```
</details>
