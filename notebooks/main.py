import decord
import numpy as np
import matplotlib.pyplot as plt
import time

from neudc.core.indexer import FPSIndexer
from neudc.core.dataloader import VideoDataloader
from neudc.core.dataset import VideoDataset

indexer = FPSIndexer('/home/alexander/msbtech/petsearch/data/14.08/19.00_до_22.00/PVN_hd_ZAO_10297_2.mp4',
                     low_fps=1, high_fps_interval=3)
indexer.max_idx, indexer.video_fps, indexer.idx_delta, indexer.high_fps_idx_interval, len(indexer.video)

dataset = VideoDataset('/home/alexander/msbtech/petsearch/data/14.08/19.00_до_22.00/PVN_hd_ZAO_10297_2.mp4', 'cpu:0')
dataloader = VideoDataloader(indexer, dataset, batch_size=4)
gen = dataloader.get_gen()


for i, (idx, batch) in enumerate(gen):
    print(idx, batch.shape)
    if i == 10:
        break
    if i == 5:
        dataloader.last_batch_info([0, 0, 1, 0])