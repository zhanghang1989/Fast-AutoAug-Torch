
from encoding.transforms.autoaug import *

def Posterize2(*args, **kwargs):
    return Posterize(*args, **kwargs)

def augment_list():  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        (CutoutAbs, 0, 20),
        (Posterize2, 0, 4),
    ]
    return l

augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Augmentation(object):
    def __init__(self, policies):
        """
        plicies : list of (name, pr, level)
        """
        self.policies = policies

    def __call__(self, img):
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() > pr:
                continue
            img = apply_augment(img, name, level)
        return img
