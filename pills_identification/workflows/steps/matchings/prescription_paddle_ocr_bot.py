from pills_identification.workflows import IWorkflowStep
from typing import List, Dict
from unidecode import unidecode
from paddleocr import PaddleOCR
import numpy as np
import torchmetrics
import json
import os
from tqdm import tqdm


class PrescriptionPaddleOCRBot(IWorkflowStep):
    def is_pill_name(self, sentence: str) -> bool:
        """Check input sentence is pill name or not
            (Example format: "1) Fabamox 500 500mg")

        Args:
            sentence str: input sentence

        Return:
            Input sentence is pill name or not
        """
        return sentence.split(")")[0].strip().isnumeric() and ")" in sentence
    
    def is_pill_time(self, sentence: str) -> bool:
        """Check input sentence is pill time or not
            (Example format: "Sáng 1 Viên")

        Args:
            sentence str: input sentence

        Return:
            
        """
        return "Sang" in sentence or "Chieu" in sentence or "Toi" in sentence

    def is_pill_amount(self, sentence: str) -> bool:
        """Check input sentence is pill time or not
            (Example format: "Sáng 1 Viên")

        Args:
            sentence str: input sentence

        Return:
            
        """
        return "SL:" in sentence

    def normalize_pill_name(self, pill_name: str) -> str:
        """Normalize pill name

        Args:
            pill_name str: pill name

        Return:
            Pill name is lowercase and no spaces
        """
        pill_name = unidecode(pill_name).strip()
        mark_start = pill_name.find(")") + 1
        mark_end = pill_name.find("sl:")

        return pill_name[mark_start:mark_end] if (mark_end != -1) else pill_name[mark_start:]

    def normalize_pill_time(self, pill_time: str) -> str:
        """Normalize pill time

        Args:
            pill_time str: pill time

        Return:
            
        """
        return pill_time.replace('Sang','Morning: ').replace('Toi','Eve: ').replace('Trua', 'Atf: ').replace('Vien','')

    def normalize_pill_amount(self, pill_amount: str) -> str:
        """Normalize pill amount

        Args:
            pill_time str: pill amount

        Return:
            
        """
        return pill_amount.replace('SL: ','').replace('Vien','')

    def filter_pill_info(self, sentences: List[str]) -> Dict:
        """Filter objects and get pill name only

        Args:
            sentences List[str]: list of input sentences

        Return:
            Dict {pill names: ABC ,amount:10, time to take: {Morning:1, Aft:1, Eve:1}}
        """
        pill_times = []
        pill_ammounts = []
        pill_names = []
        dict_pill_info = []

        for sentence in sentences:
            if self.is_pill_name(sentence):
                if len(pill_names)==0:
                    pill_names.append(self.normalize_pill_name(sentence))
                    continue
                tmp = {"Time":pill_times, "Amount":pill_ammounts}
                dict_pill_info.append({pill_names[-1]: tmp})

                pill_names.append(self.normalize_pill_name(sentence))
                pill_times = []
                pill_ammounts = []
            elif self.is_pill_time(sentence):
                pill_times.append(self.normalize_pill_time(sentence).strip())
            elif self.is_pill_amount(sentence):
                pill_ammounts.append(self.normalize_pill_amount(sentence).strip())

        tmp = {"Time":pill_times, "Amount":pill_ammounts}
        dict_pill_info.append({pill_names[-1]: tmp})
        return dict_pill_info

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
            return []

    def __call__(self, images: List[np.array], **kwargs) -> List[int]:
        """Get pill ids from prescription images

        Args:
            images List[np.array]: List of prescription images

        Returns:
            List[int]:  List of pill ids
        """
        ocr_model = PaddleOCR(lang="en", show_log=False, debug=False, use_angle_cls=True)

        list_objects = []

        for image in tqdm(images):
            list_objects.append(ocr_model.ocr(image))

        list_sentences = [[ocr_object[1][0] for ocr_object in objects] for objects in list_objects]
        list_pill_info = [self.filter_pill_info(sentences) for sentences in list_sentences]

        return {**kwargs, "pill_info": list_pill_info }