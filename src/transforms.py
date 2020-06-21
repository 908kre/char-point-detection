import itertools
import random
import numpy as np
import cv2
import torch
import albumentations as albm


class MakeMap(albm.BasicTransform):
    def __init__(self, alpha=2e-3, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.alpha = alpha

    def __call__(self, force_apply=True, **kwargs):
        sample = kwargs.copy()
        hm, size, off = make_map(
            sample["image"].shape, sample["bboxes"], alpha=self.alpha
        )
        sample["hm"] = hm
        sample["size"] = size
        sample["off"] = off
        sample["bboxes"] = np.array(sample["bboxes"])
        return sample

    def get_transform_init_args_names(self):
        return ("alpha",)


def make_map(image_shape, bboxes, alpha=2e-3):
    qh, qw = image_shape[0] // 4, image_shape[1] // 4
    grid_x, grid_y = np.meshgrid(np.arange(qw), np.arange(qh))
    hm = np.zeros(shape=(1, qh, qw), dtype=np.float32)
    size = np.zeros(shape=(2, qh, qw), dtype=np.float32)
    off = np.zeros(shape=(2, qh, qw), dtype=np.float32)
    for x, y, w, h in bboxes:
        cx, cy = (x + w / 2.0, y + h / 2.0)
        qx, qy = int(cx // 4), int(cy // 4)
        off_x, off_y = (cx % 4) / 4, (cy % 4) / 4
        sigma = np.sqrt(w ** 2 + h ** 2) * alpha
        hm_ = np.exp(-((grid_x - qx) ** 2 + (grid_y - qy) ** 2) / (2 * sigma ** 2))
        hm[0] = np.maximum(hm[0], hm_)
        size[0, qy, qx] = w
        size[1, qy, qx] = h
        off[0, qy, qx] = off_x
        off[1, qy, qx] = off_y
    return hm, size, off


class RandomDilateErode(albm.ImageOnlyTransform):
    def __init__(self, ks_limit=(1, 5), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.ks_limit = ks_limit

    def apply(self, img, ks=(1, 1), dilate=1, **params):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ks)
        if dilate:
            return cv2.dilate(img, kernel, iterations=1)
        else:
            return cv2.erode(img, kernel, iterations=1)

    def get_params(self):
        return {
            "ks": (random.randint(*self.ks_limit), random.randint(*self.ks_limit)),
            "dilate": random.randint(0, 1),
        }

    def get_transform_init_args_names(self):
        return ("ks_limit",)


class RandomRuledLines(albm.ImageOnlyTransform):
    def __init__(self, lines_limit=(30, 60), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.lines_limit = lines_limit

    def apply(
        self, img, num_lines: int, orientation: int, phase: float, color: int, **params
    ):
        img = img.copy()
        cols, rows = params["cols"], params["rows"]
        longer_side = max(cols, rows)
        num_lines = random.randint(*self.lines_limit)
        spacing = longer_side // num_lines
        thickness = int(longer_side / 2000 + 1)
        color = (color,) * 3
        pos = int(spacing * phase)
        if orientation == 0:  # horizontal
            while pos < rows:
                pt1 = (0, pos)
                pt2 = (cols, pos)
                img = cv2.line(
                    img, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_4
                )
                pos += spacing
        else:  # vertical
            while pos < cols:
                pt1 = (pos, 0)
                pt2 = (pos, rows)
                img = cv2.line(
                    img, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_4
                )
                pos += spacing
        return img

    def get_params(self):
        return {
            "num_lines": random.randint(*self.lines_limit),
            "orientation": random.randint(0, 1),
            "phase": random.random(),
            "color": random.randint(0, 64),
        }

    def get_transform_init_args_names(self):
        return ("lines_limit",)


class RandomLayout(albm.DualTransform):
    def __init__(
        self, width: int, height: int, size_limit=(0.5, 1.0), always_apply=False, p=1.0
    ):
        super().__init__(always_apply, p)
        self.width = width
        self.height = height
        self.size_limit = size_limit

    def apply(self, img, size, offset, **params):
        width = self.width * size[0]
        height = self.height * size[1]
        offset_x = self.width * offset[0]
        offset_y = self.height * offset[1]
        pts1 = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], 0]])
        pts2 = np.float32(
            [
                [offset_x, offset_y],
                [offset_x, offset_y + height],
                [offset_x + width, offset_y],
            ]
        )
        return cv2.warpAffine(
            img,
            cv2.getAffineTransform(pts1, pts2),
            (self.width, self.height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply_to_bbox(self, bbox, size, offset, **params):
        x1, y1, x2, y2 = bbox
        x1 = x1 * size[0] + offset[0]
        y1 = y1 * size[1] + offset[1]
        x2 = x2 * size[0] + offset[0]
        y2 = y2 * size[1] + offset[1]
        return x1, y1, x2, y2

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        scale = min(
            self.height / image.shape[0], self.width / image.shape[1]
        ) * random.uniform(*self.size_limit)
        size_x = image.shape[1] * scale / self.width
        size_y = image.shape[0] * scale / self.height
        offset_x = random.uniform(0, 1.0 - size_x)
        offset_y = random.uniform(0, 1.0 - size_y)
        return {"size": (size_x, size_y), "offset": (offset_x, offset_y)}

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ("width", "height", "size_limit")
