from typing import List, Dict
import torch
import numpy as np
from tqdm import tqdm

from pills_identification.workflows.pills_workflow import PillsWorkflowStep


class YOLOv5LocalizationStep(PillsWorkflowStep):
    conf_thresh = 0.45
    mIoU_thresh = 0.1
    model_path = ""

    def __init__(self, model_path, conf_thresh=0.45, mIoU_thresh=0.1, batch_size=16, **kwargs):
        """_summary_

        Raises:
            AttributeError: YOLO pretrained path was not set
        """
        super().__init__(**kwargs)

        self.conf_thresh = conf_thresh
        self.mIoU_thresh = mIoU_thresh
        self.batch_size = batch_size
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        self.model.conf = conf_thresh
        self.model.iou = mIoU_thresh

    def __call__(self, images: List[np.array], file_paths: List[str], **kwargs) -> Dict[str, List[np.array]]:
        """Crop pill from images

        Args:
            images (List[np.array]): List of images
            file_paths (List[str]): Paths of images

        Returns:
            Dict:
                images: Cropped pill
                file_paths: File path for each pill
                xyxy: Bounding box for each pill
                conf: Confidence score for each pill

        """

        batches = [images[i : i + self.batch_size] for i in range(0, len(images), self.batch_size)]

        image_results = []
        image_path_results = []
        xyxy_results = []
        conf_results = []

        for batch in tqdm(batches):
            dummy_file_paths = []

            results = self.model(batch)
            objects = results.crop(save=False)

            xyxy = results.xyxy

            for i, item in enumerate(xyxy):
                # Duplicate file_path
                dummy_file_paths.extend([file_paths[i]] * item.shape[0])

                item = item.tolist()

                # Item: List[x1,y1,x2,y2,conf,class]:
                xyxy_results.extend([data[:-2] for data in item])
                conf_results.extend([data[-2] for data in item])

            objects = [im["im"] for im in objects]

            image_results.extend(objects)
            image_path_results.extend(dummy_file_paths)

        return {
            **kwargs,
            "pill_paths": image_path_results,
            "images": image_results,
            "bounding_boxes": xyxy_results,
            "confidence_scores": conf_results,
        }
