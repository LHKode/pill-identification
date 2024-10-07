import argparse
from typing import List
import glob
import os
import cv2
import numpy as np
from pills_identification.tools.reduce_boundingbox import ReduceBoundingBox
from pills_identification.workflows.steps.image_io.opencv import OpenCVImageReader
from pills_identification.workflows.pills_workflow import PillsWorkflow
from pills_identification.workflows.steps.alignments.remove_background import DeepRemoveBackgroundStep
from pills_identification.workflows.steps.alignments.rotate_pill import RotatePillStep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input path")
    parser.add_argument("output_path", help="Output path")
    args = parser.parse_args()

    file_paths = list(glob.glob(f"{args.input_path}/*.jpg"))

    steps = [
        OpenCVImageReader(),
        DeepRemoveBackgroundStep(),
        RotatePillStep(),
        ReduceBoundingBox(),
    ]

    workflows = PillsWorkflow(steps)

    result = workflows(file_paths=file_paths)

    for image, origin_path in zip(result["images"], result["file_paths"]):
        file_name = os.path.basename(origin_path)
        new_path = os.path.join(args.output_path, f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(new_path, image)
