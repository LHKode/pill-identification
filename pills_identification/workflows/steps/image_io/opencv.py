from typing import List
import cv2
from pills_identification.workflows.pills_workflow import PillsWorkflowStep


class OpenCVImageReader(PillsWorkflowStep):
    def __call__(self, file_paths: List[str], color_mode: str = "unchanged", **kwargs):
        images = [cv2.imread(file_path, cv2.IMREAD_UNCHANGED) for file_path in file_paths]

        if color_mode == "rgb":
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

        if color_mode == "hsv":
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in images]

        return {**kwargs, "images": images, "file_paths": file_paths}
