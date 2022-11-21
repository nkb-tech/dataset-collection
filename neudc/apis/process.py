import logging
import os.path as osp
import typing as tp

import torch
import torch.nn as nn
from mmdet.apis import inference_detector
from mmdet.utils import get_root_logger

from neudc.core.dataloader import BaseDataloader, VideoDataloader
from neudc.core.dataset import CV2VideoDataset
from neudc.core.indexer import BaseIndexer
from neudc.saver import COCOSaver

# from neudc.core.preprocessing import OpticalUndistortion

# def _collate(batch_frames: np.ndarray) -> tp.List[np.ndarray]:
#     if batch_frames.ndim == 4:
#         output = [frame for frame in batch_frames]
#     elif batch_frames.ndim == 3:
#         output = [
#             frame,
#         ]
#     else:
#         raise NotImplementedError

#     return output


def process_videos(files: tp.List[str],
                   model: nn.Module,
                   batch_size: int,
                   indexer: BaseIndexer,
                   saver: COCOSaver,
                   log_file: str = None,
                   device: str = 'cpu:2',
                   num_threads: int = 0) -> None:

    logger = get_root_logger(log_file=log_file, log_level=logging.INFO)

    num_files = len(files)

    for i in range(num_files):

        file_path = files[i]
        logger.info(f'{i}/{num_files}: {file_path}')

        # preproc = \
        # OpticalUndistortion('configs/cam_mtx/cam_mat_PVN_hd_ZAO_10297_2.npy',
        # 'configs/cam_mtx/cam_dist_PVN_hd_ZAO_10297_2.npy')
        dataset = CV2VideoDataset(video_path=file_path,
                                  device=device,
                                  num_threads=num_threads)
        #   preprocessing=preproc)
        # set indexer
        indexer.set_video(max_idx=dataset.max_idx, video_fps=dataset.video_fps)
        # set saver
        saver.set_video(osp.basename(file_path))
        # set dataloader
        dataloader = VideoDataloader(indexer=indexer,
                                     dataset=dataset,
                                     batch_size=batch_size)

        # clear cuda cache
        if isinstance(device, torch.device) and \
           device.type == 'cuda' and \
           torch.cuda.is_available():
            torch.cuda.empty_cache()

        process_video(model=model,
                      dataloader=dataloader,
                      saver=saver,
                      log_file=log_file)

        saver.dump()

    logger.info(f'Tasks {num_files}/{num_files} have done.')


def process_video(model: nn.Module,
                  dataloader: BaseDataloader,
                  saver: COCOSaver,
                  log_file: str = None) -> None:

    logger = get_root_logger(log_file=log_file, log_level=logging.INFO)

    # for each image in buffer
    # dataloader has no len
    for batch_idxs, batch_frames in dataloader.get_gen():
        # batch_frames = _collate(batch_frames)
        # call forward
        batch_preds = inference_detector(model, batch_frames)
        # filter predictions and save
        batch_filter_mask = saver(batch_frames, batch_preds, batch_idxs)
        # update dataloader mask
        dataloader.last_batch_info(batch_filter_mask)
        # TODO write more understandable stats
        logger.info(f'Process {batch_idxs}/{dataloader.dataset.max_idx}, '
                    f'{max(batch_idxs)/dataloader.dataset.max_idx*100:2.1f}%, '
                    f'found {sum(batch_filter_mask)} objects.')
