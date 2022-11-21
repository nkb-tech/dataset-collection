from typing import Union

import cv2
import numpy as np

# This module contains preprocessing classes for images.
# Those classes must have __call__ methods implemented, that
# take an image as an input and output an image or torch.tensor,
# or a numpy.ndarray. Classes have to be similar to torchvision.transforms.
# Frame undistortion has to be implemented here as a function.


class OpticalUndistortion():
    def __init__(self, mtx_path: str, dist_path: str):
        self.mtx_path = mtx_path
        self.dist_path = dist_path
        with open(mtx_path, 'rb') as f:
            self.mtx = np.load(f)
        with open(dist_path, 'rb') as f:
            self.dist = np.load(f)
        self.h, self.w = None, None
        self.newcameramtx = None

    def _process_img(self, img: np.ndarray):
        return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)

    def _set_newcammtx(self, imgs):
        self._check_sizes(imgs)
        self.newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (self.w, self.h), 0, (self.w, self.h))

    def _check_sizes(self, imgs):
        if type(imgs) is list:
            sizes = np.asarray([[img.shape[0], img.shape[1]] for img in imgs])
            if np.any(sizes[:, 0] - sizes[0, 0]) or np.any(sizes[:, 1] -
                                                           sizes[0, 1]):
                raise ValueError(
                    f'Different image sizes, images must be the same size.'
                    f'{sizes}')
            if self.h is None and self.w is None:
                self.h, self.w, _ = imgs[0].shape
            else:
                if (self.h, self.w) != imgs[0].shape[:2]:
                    raise ValueError(
                        f'Different image sizes, images must be the same size.'
                        f'Old size: {self.h,self.w}, new size: {imgs[0].shape}'
                    )
        else:
            if self.h is None and self.w is None:
                self.h, self.w, _ = imgs.shape
            else:
                if (self.h, self.w) != imgs.shape[:2]:
                    raise ValueError(
                        f'Different image sizes, images must be the same size.'
                        f'Old size: {self.h, self.w}, new size: {imgs.shape}')
        return self.h, self.w

    def __call__(self, imgs: Union[list, np.ndarray]):
        if self.newcameramtx is None:
            self._set_newcammtx(imgs)
        self._check_sizes(imgs)
        if type(imgs) is list:
            res = [self._process_img(img) for img in imgs]
        else:
            res = [self._process_img(imgs)]
        return res
