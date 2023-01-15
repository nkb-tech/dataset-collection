from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu, gpu


class BaseDataset(metaclass=ABCMeta):
    def __init__(self,
                 video_path: str,
                 preprocessing: Optional[Callable] = None) -> None:
        self.video_path = video_path
        self.preprocessing = preprocessing

    @abstractmethod
    def __call__(self, idx: int) -> Union[np.ndarray, torch.tensor]:
        '''
        Returns video frame for current idx after preprocessing
        '''
        pass


class DecordVideoDataset(BaseDataset):
    '''
    Working incorrectly when reading frames with step > 5.
    '''
    def __init__(self,
                 video_path: str,
                 device: str,
                 num_threads: int = 0,
                 preprocessing: Optional[Callable] = None) -> None:
        super().__init__(video_path, preprocessing)
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
                raise IndexError(f'Indices {idx} is out of video '
                                 f'range (max_idx={self.max_idx})')
            frames = self.video.get_batch(idx).asnumpy()
        else:
            if idx >= self.max_idx:
                raise IndexError(f'Index {idx} is out of video '
                                 f'range (max_idx={self.max_idx})')
            frames = self.video[idx].asnumpy()

        if self.preprocessing is not None:
            return self.preprocessing(frames)
        else:
            return self.bgr2rgb(frames)


class CV2VideoDataset(BaseDataset):
    def __init__(self,
                 video_path: str,
                 preprocessing: Optional[Callable] = None) -> None:
        super().__init__(video_path, preprocessing)
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise ValueError(f'Video {video_path} is not opened')
        self.max_idx = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)

    def _get_frame(self, idx):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = self.video.read()
        if not success:
            print(f'Frame {idx} reading from file '
                  f'{self.video_path} returned an error')
            self.video.release()
            return None
        return frame

    def __call__(self, idx: Union[int, list]) -> List[np.ndarray]:
        '''
        If idx is a list if indices, then:
        - It must be ordered in ascending way
        - If there are indices, close to or bigger than self.max_idx,
            then after reading the last idx with error,
            the dataset video file is closed
            and the dataset can not be reused
        '''

        if type(idx) is list:
            frames = []
            for i in idx:
                if i >= self.max_idx:
                    continue
                frame = self._get_frame(i)
                if frame is not None:
                    frames.append(frame)
                else:
                    break
        else:
            frame = self._get_frame(idx)
            if frame is not None:
                frames = [frame]
        # frames = self.bgr2rgb(frames)
        if self.preprocessing is not None:
            frames = self.preprocessing(frames)
        return frames
