from abc import ABCMeta, abstractmethod
import numpy as np
from decord import VideoReader, cpu

class BaseIndexer(metaclass=ABCMeta):
    def __init__(self,
                 low_fps: float,
                 high_fps_interval: float) -> None:
        '''
        low_fps: fps to use while looking through video
        high_fps_interval: interval in seconds to look around the frame, 
            where the target class was found.
        '''
        self.low_fps = low_fps
        self.high_fps_interval = high_fps_interval
        self.current_idx = 0
    
    def set_video(self,
                  max_idx: int,
                  video_fps: float) -> None:
        self.max_idx = max_idx
        self.video_fps = video_fps
     
    @abstractmethod
    def idx_gen(self):
        '''
        Yields next idx for a video. 
        Can implement motion detection, looking for next suitable video frame.
        Changes current_idx attribute.
        '''
        for i in range(self.max_idx):
            local_current = self.current_idx
            yield self.current_idx
            if self.current_idx == local_current:
                self.current_idx += self.low_fps
    
    @abstractmethod
    def get_idx_around_target(self, idx):
        '''
        Generates idx list with video fps around the frame with target object in high_fps_interval.
        Changes self.current_idx not to reprocess same idx again.
        '''
        lst = np.arange(idx-10, idx+10, 1)
        self.current = idx+10
        return lst


class FPSIndexer(BaseIndexer):
    '''
    Stable version
    '''
    def __init__(self,
                 low_fps: float,
                 high_fps_interval: float) -> None:
        '''
        low_fps: fps to use while looking through video
        high_fps_interval: interval in seconds to look around the frame, 
            where the target class was found.
        '''
        super().__init__(low_fps, high_fps_interval)

    def set_video(self,
                  max_idx: int,
                  video_fps: float) -> None:
        self.max_idx = max_idx
        self.video_fps = video_fps
        self.idx_delta = self.video_fps / self.low_fps
        self.high_fps_idx_interval = int(self.video_fps * self.high_fps_interval)
    
    def idx_gen(self):
        '''
        Yields next idx for a video. 
        Can implement motion detection, looking for next suitable video frame.
        Changes current_idx attribute.
        '''
        self.processed_idx = set()
        self.current_idx = 0
        self.current_idx_float = 0
        while self.current_idx < self.max_idx:
            local_current = self.current_idx
            self.processed_idx.add(local_current)
            yield local_current
            if self.current_idx == local_current:
                self.current_idx_float += self.idx_delta
                self.current_idx = int(self.current_idx_float)
            while self.current_idx in self.processed_idx:
                self.current_idx_float += self.idx_delta
                self.current_idx = int(self.current_idx_float)
    
    def get_idx_around_target(self, idx):
        '''
        Generates idx list with video fps around the frame with target object in high_fps_interval.
        Changes self.current_idx not to reprocess same idx again.
        '''
        min_border = max(0, int(idx - self.high_fps_idx_interval))
        max_border = min(int(idx + self.high_fps_idx_interval), self.max_idx)
        lst = np.asarray([i for i in range(min_border, max_border, 1) if i not in self.processed_idx], dtype=int)
        self.current_idx_float = max(max_border, self.current_idx_float)
        self.current_idx = int(self.current_idx_float)
        self.processed_idx.update(lst)
        return lst

