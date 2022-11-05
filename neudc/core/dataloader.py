from abc import ABCMeta, abstractmethod
from typing import Union
from queue import Queue
import numpy as np
import torch
from .indexer import BaseIndexer
from .dataset import BaseDataset

class BaseDataloader(metaclass=ABCMeta):
    def __init__(self, 
                 indexer: BaseIndexer,
                 dataset: BaseDataset,
                 batch_size: int) -> None:
        '''
        indexer: video indexer
        dataset: video dataset
        batch_size: int
        device: torch device (cuda/cpu)
        '''
        self.indexer = indexer
        self.dataset = dataset
        self.batch_size = batch_size
     
    def __iter__(self):
        return self
    
    # @abstractmethod
    # def __next__(self):
    #     pass

    @abstractmethod
    def last_batch_info(self, mask: Union[list, np.ndarray]):
        '''
        mask: boolean/[0,1] mask of targetr object presence indicators in the last batch
        '''
        pass


class VideoDataloader(BaseDataloader):
    def __init__(self, 
                 indexer: BaseIndexer,
                 dataset: BaseDataset,
                 batch_size: int) -> None:
        '''
        indexer: video indexer
        dataset: video dataset
        batch_size: int
        device: torch device (cuda/cpu)
        '''
        super().__init__(indexer, dataset, batch_size)
        self.idx_gen = self.indexer.idx_gen()
        self.mask = [0] * self.batch_size
    
    def get_gen(self):
        '''
        Yields a batch of video frames
        '''
        last_batch_idx = []
        
        for i, idx in enumerate(self.idx_gen):
            last_batch_idx.append(idx)
            if len(last_batch_idx) == self.batch_size:
                yield last_batch_idx, self.dataset(last_batch_idx)
                idx_with_target = [j for j, indicator in zip(last_batch_idx, self.mask) if indicator == True]
                q = Queue(maxsize=self.indexer.high_fps_idx_interval * 2 * self.batch_size)
                for j in idx_with_target:
                    list(map(q.put, self.indexer.get_idx_around_target(j)))  
                while q.qsize() >= self.batch_size:
                    local_batch_idx = []
                    for _ in range(self.batch_size):
                        local_batch_idx.append(q.get())
                    yield local_batch_idx, self.dataset(local_batch_idx)
                local_batch_idx = []
                while not q.empty():
                    local_batch_idx.append(q.get())
                if len(local_batch_idx) > 0:
                    yield local_batch_idx, self.dataset(local_batch_idx)
                last_batch_idx = []


    def last_batch_info(self, mask: Union[list, np.ndarray]):
        '''
        mask: boolean/[0,1] mask of targetr object presence indicators in the last batch
        '''
        self.mask = mask
        