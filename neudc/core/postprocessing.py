import torch
from typing import Tuple, List, Dict

def select_instances(
    predicted_instances: Tuple[torch.Tensor],
    target_classes: List[str] = [
        'dog',
    ],
    threshold_confidences: List[float] = [
        1e-2,
    ],
    idx_mapping: Dict[int, str] = idx_mapping,
) -> Tuple[torch.Tensor]:
    '''
    Selects target classes above threshold confidence
    from output mmdetection model
    Args:
        predicted_instances (Tuple[Tensor]):
        target_classes (List[str]):
        threshold_confidences (List[float]):
        idx_mapping (Dict[int, str]):
    Returns:
        Tuple[Tensor]: 
    '''

    # TODO ins_results can be only in segmentator models
    bboxes, masks = predicted_instances['ins_results']

    target_idxs = [
        idx_mapping[target_class]
        for target_class in target_classes
    ]
    if isinstance(threshold_confidences, float):
        threshold_confidences = [threshold_confidences] * len(target_idxs)

    target_bboxes, target_masks, filter_mask = [], [], []

    for target_idx, target_treshold_confidence in zip(target_idxs,
                                                      threshold_confidences):
        is_object_here = False
        filtered_bboxes, filtered_masks = [], []
        for bbox, mask in zip(bboxes[target_idx], masks[target_idx]):
            if len(bbox) == 0:
                continue
            x_l, y_l, x_r, y_r, sc = bbox
            if sc > target_treshold_confidence:
                filtered_bboxes.append(bbox)
                filtered_masks.append(mask)
                is_object_here = True
        target_bboxes.append(np.array(filtered_bboxes))
        target_masks.append(np.array(filtered_masks))
        filter_mask.append(is_object_here)

    target_result = (target_bboxes, target_masks, filter_mask)

    return target_result
