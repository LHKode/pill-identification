from pills_identification.workflows import IWorkflowStep
from typing import List, Dict
from unidecode import unidecode
from paddleocr import PaddleOCR
import numpy as np
import torchmetrics
import json
import os
from tqdm import tqdm


class PrescriptionPaddleOCR(IWorkflowStep):
    def is_pill_name(self, sentence: str) -> bool:
        """Check input sentence is pill name or not
            (Example format: "1) Fabamox 500 500mg")

        Args:
            sentence str: input sentence

        Return:
            Input sentence is pill name or not
        """
        return sentence.split(")")[0].strip().isnumeric() and ")" in sentence

    def normalize_pill_name(self, pill_name: str) -> str:
        """Normalize pill name

        Args:
            pill_name str: pill name

        Return:
            Pill name is lowercase and no spaces
        """
        pill_name = unidecode(pill_name).lower().replace(" ", "")

        mark_start = pill_name.find(")") + 1
        mark_end = pill_name.find("sl:")

        return pill_name[mark_start:mark_end]

    def filter_pill_name(self, sentences: List[str]) -> List[str]:
        """Filter objects and get pill name only

        Args:
            sentences List[str]: list of input sentences

        Return:
            List of pill names
        """
        return [self.normalize_pill_name(sentence) for sentence in sentences if self.is_pill_name(sentence)]

    def get_pill_id(self, pill_name: str, mapping: Dict[str, int]) -> int:
        """Get truth pill names with minimum CharErrorRate to pill names

        Args:
            pill_names str: input pill name
            mapping_pill_names List[str]: list of truth pill names

        Return:
            similar pill name
        """
        metric = torchmetrics.CharErrorRate()
        min_error = 1
        similar_pill_name = ""

        for truth_pill_name in mapping.keys():
            if min_error > metric(pill_name, truth_pill_name):
                min_error = metric(pill_name, truth_pill_name)
                similar_pill_name = truth_pill_name

        try:
            return mapping[similar_pill_name]
        except Exception as e:
            print(e)
            print("ERR ", pill_name)
            return []

    def __call__(self, images: List[np.array], **kwargs) -> List[int]:
        """Get pill ids from prescription images

        Args:
            images List[np.array]: List of prescription images

        Returns:
            List[int]:  List of pill ids
        """
        ocr_model = PaddleOCR(lang="en", show_log=False, debug=False, use_angle_cls=True)

        print("START OCR")

        list_objects = []

        for image in tqdm(images):
            list_objects.append(ocr_model.ocr(image))

        # list_objects = [ocr_model.ocr(image) for image in images]

        print("DONE OCR")

        list_sentences = [[ocr_object[1][0] for ocr_object in objects] for objects in list_objects]

        list_pill_names = [self.filter_pill_name(sentences) for sentences in list_sentences]

        script_dir = os.path.dirname(__file__)
        mapping_path = "drug_id_mapping.json"
        abs_mapping_path = os.path.join(script_dir, mapping_path)
        with open(abs_mapping_path) as f:
            mapping = json.load(f)

        list_pill_ids = []
        for pill_names in list_pill_names:
            pill_ids = []
            for pill_name in pill_names:
                temp = self.get_pill_id(pill_name, mapping)

                print(temp)
                pill_ids.extend(self.get_pill_id(pill_name, mapping))
            list_pill_ids.append(pill_ids)

        get_file_name = lambda x: x[:-4].replace("_TEST", "")

        file_names = [get_file_name(file_path.split("/")[-1]) for file_path in kwargs["file_paths"]]

        dict_pill_ids = dict(zip(file_names, list_pill_ids))

        return {**kwargs, "prescription_ids": dict_pill_ids}
