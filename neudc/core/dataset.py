from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union

import numpy as np
import torch
from decord import VideoReader, cpu, gpu


class BaseDataset(metaclass=ABCMeta):
    def __init__(self,
                 video_path: str,
                 device: str,
                 preprocessing: Optional[Callable] = None) -> None:
        self.video_path = video_path
        self.preprocessing = preprocessing
        self.device = device

    @abstractmethod
    def __call__(self, idx: int) -> Union[np.ndarray, torch.tensor]:
        '''
        Returns video frame for current idx after preprocessing
        '''
        pass


class VideoDataset(BaseDataset):
    def __init__(self,
                 video_path: str,
                 device: str,
                 num_threads: int = 0,
                 preprocessing: Optional[Callable] = None) -> None:
        super().__init__(self, video_path, preprocessing)
        dev, num = device.split(':')
        num = int(num)
        if dev == 'cpu':
            decord_device = cpu(num)
        elif dev == 'cuda':
            decord_device = gpu(num)
        else:
            raise ValueError(f'Unknown device type: {device}')
        self.video = VideoReader(video_path,
                                 ctx=decord_device,
                                 num_threads=num_threads)
        self.max_idx = len(self.video)
        self.video_fps = self.video.get_avg_fps()

    def bgr2rgb(self, data: np.ndarray) -> np.ndarray:
        return data[..., ::-1]

    def __call__(self, idx: Union[int,
                                  list]) -> Union[np.ndarray, torch.tensor]:
        if type(idx) is list:
            if max(idx) >= self.max_idx:
                raise IndexError(
                    f'Indices {idx} is out of video range (max_idx={self.max_idx})'
                )
            frames = self.video.get_batch(idx).asnumpy()
        else:
            if idx >= self.max_idx:
                raise IndexError(
                    f'Index {idx} is out of video range (max_idx={self.max_idx})'
                )
            frames = self.video[idx].asnumpy()

        if self.preprocessing is not None:
            return self.preprocessing(frames)
        else:
            return self.bgr2rgb(frames)
