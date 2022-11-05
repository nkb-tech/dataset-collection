from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Union
import torch
import numpy as np
from decord import VideoReader, cpu, gpu

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, 
                 video_path: str,
                 device: str,
                 preprocessing: Optional[Callable]=None) -> None:
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
                 preprocessing: Optional[Callable]=None) -> None:
        super().__init__(self, video_path, preprocessing)
        dev, num = device.split(':')
        num = int(num)
        if dev == 'cpu':
            decord_device = cpu(num)
        elif dev == 'cuda':
            decord_device = gpu(num)
        else:
            raise ValueError(f'Unknown device type: {device}')
        self.video = VideoReader(video_path, ctx=decord_device)
        self.max_idx = len(self.video)
    
    def __call__(self, idx: Union[int, list]) -> Union[np.ndarray, torch.tensor]:
        if type(idx) is list:
            if max(idx) >= self.max_idx:
                raise IndexError(f'Indices {idx} is out of video range (max_idx={self.max_idx})')
            frames = self.video.get_batch(idx).asnumpy()
        else:
            if idx >= self.max_idx:
                raise IndexError(f'Index {idx} is out of video range (max_idx={self.max_idx})')
            frames = self.video[idx].asnumpy()
        
        if self.preprocessing is not None:
            return self.preprocessing(frames)
        else:
            return frames
        
        
    