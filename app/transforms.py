from typing import Any, Tuple, List, Dict
import itertools
from os import remove
import random
import numpy as np
import cv2
import torch
import albumentations as albm


class FilterBbox(albm.BasicTransform):
    def __init__(self, always_apply: bool = False, p: float = 1.0) -> None:
        super().__init__(always_apply, p)

    def __call__(self, force_apply: bool = True, **kwargs: Any) -> Any:
        sample = kwargs.copy()
        rows, cols = sample["image"].shape[:2]
        bboxes = sample["bboxes"]
        num_boxes = len(bboxes)
        bboxes = np.array(bboxes) if num_boxes > 0 else np.zeros((0, 4))
        labels1 = np.array(sample["labels1"])
        labels2 = np.array(sample["labels2"])

        bboxes[:, 2:] += bboxes[:, :2]  # coco to pascal
        eps = 1e-6
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, cols - eps)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, rows - eps)
        bboxes[:, 2:] -= bboxes[:, :2]  # pascal to coco

        mask = (bboxes[:, 2] >= 8) & (bboxes[:, 3] >= 8) & (labels1 == 1)

        sample["bboxes"] = bboxes[mask]
        sample["labels1"] = labels1[mask]
        sample["labels2"] = labels2[mask]
        return sample


class RandomDilateErode(albm.ImageOnlyTransform):
    def __init__(
        self,
        ks_limit: Tuple[int, int] = (1, 5),
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.ks_limit = ks_limit

    def apply(
        self, img: Any, ks: Tuple[int, int] = (1, 1), dilate: int = 1, **params: Any
    ) -> Any:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ks)
        if dilate:
            return cv2.dilate(img, kernel, iterations=1)
        else:
            return cv2.erode(img, kernel, iterations=1)

    def get_params(self) -> Dict:
        return {
            "ks": (random.randint(*self.ks_limit), random.randint(*self.ks_limit)),
            "dilate": random.randint(0, 1),
        }

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("ks_limit",)


class RandomRuledLines(albm.ImageOnlyTransform):
    def __init__(
        self,
        lines_limit: Tuple[int, int] = (30, 60),
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply, p)
        self.lines_limit = lines_limit

    def apply(
        self,
        img: Any,
        num_lines: int,
        orientation: int,
        phase: float,
        color: int,
        **params: Any
    ) -> Any:
        img = img.copy()
        cols, rows = params["cols"], params["rows"]
        longer_side = max(cols, rows)
        num_lines = random.randint(*self.lines_limit)
        spacing = longer_side // num_lines
        thickness = int(longer_side / 2000 + 1)
        _color = (color,) * 3
        pos = int(spacing * phase)
        if orientation == 0:  # horizontal
            while pos < rows:
                pt1 = (0, pos)
                pt2 = (cols, pos)
                img = cv2.line(
                    img, pt1, pt2, _color, thickness=thickness, lineType=cv2.LINE_4
                )
                pos += spacing
        else:  # vertical
            while pos < cols:
                pt1 = (pos, 0)
                pt2 = (pos, rows)
                img = cv2.line(
                    img, pt1, pt2, _color, thickness=thickness, lineType=cv2.LINE_4
                )
                pos += spacing
        return img

    def get_params(self) -> Dict:
        return {
            "num_lines": random.randint(*self.lines_limit),
            "orientation": random.randint(0, 1),
            "phase": random.random(),
            "color": random.randint(0, 64),
        }

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("lines_limit",)


class RandomLayout(albm.DualTransform):
    def __init__(
        self,
        width: int,
        height: int,
        size_limit: Tuple[float, float] = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.width = width
        self.height = height
        self.size_limit = size_limit

    def apply(
        self, img: Any, size: Tuple[int, int], offset: Tuple[int, int], **params: Any
    ) -> Any:
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
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply_to_bbox(
        self, bbox: Any, size: Tuple[int, int], offset: Tuple[int, int], **params: Any
    ) -> Any:
        x1, y1, x2, y2 = bbox
        x1 = x1 * size[0] + offset[0]
        y1 = y1 * size[1] + offset[1]
        x2 = x2 * size[0] + offset[0]
        y2 = y2 * size[1] + offset[1]
        return x1, y1, x2, y2

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Any) -> Dict:
        image = params["image"]
        scale = min(
            self.height / image.shape[0], self.width / image.shape[1]
        ) * random.uniform(*self.size_limit)
        size_x = image.shape[1] * scale / self.width
        size_y = image.shape[0] * scale / self.height
        offset_x = random.uniform(0, 1.0 - size_x)
        offset_y = random.uniform(0, 1.0 - size_y)
        return {
            "size": (size_x, size_y),  # transformed image size
            "offset": (offset_x, offset_y),
        }

    def get_params(self) -> Dict:
        return {}

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("width", "height", "size_limit")


class RandomBinarize(albm.ImageOnlyTransform):
    def __init__(self, always_apply: float = False, p: float = 0.5) -> None:
        super().__init__(always_apply, p)

    def apply(self, img: Any, **params: Any) -> Any:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        bs = random.randint(2, 5) * 2 + 1
        c = random.randint(5, 12)
        binimg = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, c
        )
        return np.stack([binimg] * 3, axis=-1)

    def get_params(self) -> Dict:
        return {}

    def get_transform_init_args_names(self) -> Tuple:
        return ()


class LongestMaxSizeWithDistort(albm.DualTransform):
    def __init__(
        self,
        max_size: int = 1024,
        distort_lim: float = 0.1,
        interpolation: Any = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size
        self.distort_lim = distort_lim

    def apply(self, img: Any, distort: Any, **params: Any) -> Any:
        ratio = img.shape[0] / img.shape[1] * distort
        if ratio > 1.0:
            rows = self.max_size
            cols = int(rows / ratio)
        else:
            cols = self.max_size
            rows = int(cols * ratio)
        return cv2.resize(img, (cols, rows), interpolation=self.interpolation)

    def apply_to_bbox(self, bbox: Any, **params: Any) -> Any:
        # Bounding box coordinates are scale invariant
        return bbox

    def get_params(self) -> Dict:
        return {
            "distort": random.uniform(1.0 - self.distort_lim, 1.0 + self.distort_lim)
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("max_size", "distort_lim", "interpolation")


class RandomCrop(albm.BasicTransform):
    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        always_apply: bool = True,
        p: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height

    def __call__(self, force_apply: bool = True, **kwargs: Any) -> Any:
        sample = kwargs.copy()
        image = sample["image"]
        bboxes = sample["bboxes"]
        rows, cols, _ = image.shape
        origin = self.get_origin(rows, cols)
        sample["image"] = self.apply_to_image(image, origin)
        sample["bboxes"], removed = self.apply_to_bboxes(bboxes, origin)
        sample["labels1"] = [
            _ for i, _ in enumerate(sample["labels1"]) if i not in removed
        ]
        sample["labels2"] = [
            _ for i, _ in enumerate(sample["labels2"]) if i not in removed
        ]
        return sample

    def get_origin(self, rows: int, cols: int) -> Tuple[int, int]:
        range_x = sorted([0, cols - self.width])
        range_y = sorted([0, rows - self.height])
        x = random.uniform(*range_x)
        y = random.uniform(*range_y)
        return int(x), int(y)

    def apply_to_image(self, image: Any, origin: Tuple[int, int]) -> Any:
        src_x, src_y = origin
        src_w, src_h = self.width, self.height
        dst_x, dst_y, dst_w, dst_h = 0, 0, self.width, self.height
        pts1 = np.float32(
            [[src_x, src_y], [src_x + src_w, src_y], [src_x, src_y + src_h]]
        )
        pts2 = np.float32(
            [[dst_x, dst_y], [dst_x + dst_w, dst_y], [dst_x, dst_y + dst_h]]
        )
        return cv2.warpAffine(
            image,
            cv2.getAffineTransform(pts1, pts2),
            (self.width, self.height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply_to_bboxes(self, bboxes: Any, origin: Tuple[int, int]) -> Any:
        origin_x, origin_y = origin
        applied = []
        removed = []
        for idx, bbox in enumerate(bboxes):
            x1, y1, w, h = bbox

            # shift
            x1, y1 = x1 - origin_x, y1 - origin_y

            # remove outside
            xc, yc = x1 + w / 2, y1 + h / 2
            if xc < 0 or xc >= self.width or yc < 0 or yc >= self.height:
                removed.append(idx)
                continue

            # crop box
            x2, y2 = x1 + w, y1 + h
            x1, x2 = np.clip([x1, x2], 0, self.width)
            y1, y2 = np.clip([y1, y2], 0, self.height)
            w, h = x2 - x1, y2 - y1

            # remove small box
            if w < 4 or h < 4:
                removed.append(idx)
                continue

            applied.append((x1, y1, w, h))

        return applied, removed

    def get_transform_init_args_names(self) -> List[str]:
        return ["width", "height"]
