import logging
import torch.nn as nn
from mmdet.apis import inference_detector
from neudc.core.dataloader import BaseDataloader
from neudc.core.indexer import BaseIndexer
from neudc.core.dataset import BaseDataset
from neudc.saver import COCOSaver

def process_video(model: nn.Module,
                  dataloader: BaseDataloader,
                  saver: COCOSaver,
                  logger: logging.Logger) -> None:

    # for each image in buffer
    # dataloader has no len
    for batch_idxs, batch_frames in dataloader.get_gen():
        # call forward
        batch_preds = inference_detector(model, batch_frames)
        # filter predictions and save
        batch_filter_mask = saver(batch_frames,
                                  batch_preds,
                                  batch_idxs)
        # update dataloader mask
        dataloader.last_batch_info(batch_filter_mask)
        # TODO write more understandable stats
        logger.info(f'Process {batch_idxs}.')

    saver.dump()
