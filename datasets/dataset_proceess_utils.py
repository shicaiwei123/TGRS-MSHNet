import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as tt
import torchvision.transforms.functional as F
import pdb
from PIL import Image


class RandomHorizontalFlip_multi(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        if random.random() < self.p:
            for index in range(len(keys) - 1):
                value = sample[keys[index]]
                sample[keys[index]] = cv2.flip(value, 1)
            return sample
        else:
            return sample


class RandomVerticalFlip_multi(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        if random.random() < self.p:
            for index in range(len(keys) - 1):
                value = sample[keys[index]]
                sample[keys[index]] = cv2.flip(value, 0)
            return sample
        else:
            return sample

class Resize_multi(object):

    def __init__(self, size):
        '''
        元组size,如(112,112)
        :param size:
        '''
        self.size = size

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            sample[keys[index]] = cv2.resize(value, self.size)
        return sample


class RondomRotion_multi(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)
        value_temp = sample[keys[0]]

        (h, w) = value_temp.shape[:2]
        (cx, cy) = (w / 2, h / 2)

        # 设置旋转矩阵
        angle = random.randint(-self.angle, self.angle)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0]) * 0.8
        sin = np.abs(M[0, 1]) * 0.8

        # 计算图像旋转后的新边界
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            sample[keys[index]] = cv2.warpAffine(value, M, (nw, nh))
        return sample


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ColorAdjust_multi(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast)
        self.saturation = self._check_input(saturation)
        self.hue = [0 - hue, 0 + hue]

    def _check_input(self, value, center=1, clip_first_on_zero=True):

        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = tt.Compose(transforms)

        return transform

    def __call__(self, sample):

        keys = sample.keys()
        keys = list(keys)
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
            value_pil = Image.fromarray(value)
            value_tramsform = transform(value_pil)
            value = np.array(value_tramsform)
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
            sample[keys[index]] = value
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RondomCrop_multi(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)
        value_temp = sample[keys[0]]

        h, w = value_temp.shape[:2]
        # print(h,w,self.size,h - self.size)

        y = np.random.randint(0, h - self.size)
        x = np.random.randint(0, w - self.size)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            sample[keys[index]] = value[y:y + self.size, x:x + self.size, :]
        return sample


class Cutout_multi(object):
    '''
    作用在to tensor 之后
    '''

    def __init__(self, length=30):
        self.length = length

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)
        value_temp = sample[keys[0]]

        h, w = value_temp.shape[1], value_temp.shape[2]  # Tensor [1][2],  nparray [0][1]
        length_new = np.random.randint(1, self.length)
        y = np.random.randint(h - length_new)
        x = np.random.randint(w - length_new)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value[y:y + length_new, x:x + length_new] = 0
            sample[keys[index]] = value
        return sample


class Normaliztion_multi(object):
    """

    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        keys = sample.keys()
        keys = list(keys)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            min_values = torch.min(value, dim=2, keepdim=True)[0]
            min_values = torch.min(min_values, dim=1, keepdim=True)[0]
            max_values = torch.max(value, dim=2, keepdim=True)[0]
            max_values = torch.max(max_values, dim=1, keepdim=True)[0]
            value = (value - min_values) / (max_values - min_values)
            sample[keys[index]] = value

        return sample


class ToTensor_multi(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W

        keys = sample.keys()
        keys = list(keys)

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            if len(value.shape) == 2:
                value = np.expand_dims(value, axis=2)
            value = value.transpose((2, 0, 1))
            sample[keys[index]] = value

        for index in range(len(keys) - 1):
            value = sample[keys[index]]
            value = np.array(value)
            value = torch.from_numpy(value.astype(np.float)).float()
            # print(value.type())
            sample[keys[index]] = value

        return sample
