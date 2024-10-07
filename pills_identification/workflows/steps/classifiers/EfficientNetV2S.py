from typing import List, Dict
from tensorflow import keras
import tensorflow as tf
import numpy as np
import json
import cv2
import os

from pills_identification.workflows.pills_workflow import PillsWorkflowStep

class EfficientNetV2SClassificationStep(PillsWorkflowStep):
    model_path = ""

    def __init__(self, model_path: str, **kwargs):
        """_summary_

        Raises:
            AttributeError: Model pretrained path was not set
        """
        super().__init__(**kwargs)
        self.model = keras.models.load_model(model_path)

    def __call__(self, images: List[np.array], **kwargs) -> Dict[str, List[int]]:
        """Classify pillIDs from images

        Args:
            images (List[np.array]): List of images

        Returns:
            Dict:
                class: List pillids of images
                conf: Confidence score for each pillids
        """
        image_height = 180
        image_width = 180


        predictions = [
            self.model.predict(tf.expand_dims(cv2.resize(image, dsize=(image_height, image_width)), 0))
            for image in images
        ]
        scores = [tf.nn.softmax(prediction[0]) for prediction in predictions]

        script_dir = os.path.dirname(__file__)
        mapping_path = "class_mapping.json"
        abs_mapping_path = os.path.join(script_dir, mapping_path)
        with open(abs_mapping_path) as f:
            class_mapping = json.load(f)

        pillids = [class_mapping[str(np.argmax(score))] for score in scores]
        scores = [np.max(score) for score in scores]

        return {
            **kwargs,
            "pill_ids": pillids,
            "confidence_scores": scores,
        }
