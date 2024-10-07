import argparse
from typing import List
import glob
import os
import cv2
import numpy as np
from pills_identification.workflows.steps.image_io.opencv import OpenCVImageReader
from pills_identification.workflows.pills_workflow import PillsWorkflow, PillsWorkflowStep


class ReduceBoundingBox(PillsWorkflowStep):
    def __call__(self, images: List[np.array], **kwargs):
        return {**kwargs, "images": [self.reduce_bounding_box(image) for image in images]}

    @staticmethod
    def reduce_bounding_box(img: np.array):
        h, w, _ = img.shape

        top, left, bottom, right = 0, 0, 0, 0

        sum_rows = [np.sum(img[i, :, 3] > 50) for i in range(h)]
        sum_cols = [np.sum(img[:, i, 3] > 50) for i in range(w)]

        for i in range(h):
            if sum_rows[i] != 0:
                top = i
                break

        for i in reversed(range(h)):
            if sum_rows[i] != 0:
                bottom = i
                break

        for i in range(w):
            if sum_cols[i] != 0:
                left = i
                break

        for i in reversed(range(w)):
            if sum_cols[i] != 0:
                right = i
                break

        if left < right and top < bottom:
            return img[top:bottom,left: right, :]

        return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input path")
    parser.add_argument("output_path", help="Output path")
    args = parser.parse_args()

    file_paths = list(glob.glob(f"{args.input_path}/*.png"))

    steps = [
        OpenCVImageReader(),
        ReduceBoundingBox(),
    ]
    workflows = PillsWorkflow(steps)

    result = workflows(file_paths=file_paths)

    for image, origin_path in zip(result["images"], result["file_paths"]):
        file_name = os.path.basename(origin_path)
        new_path = os.path.join(args.output_path, file_name)
        cv2.imwrite(new_path, image)
