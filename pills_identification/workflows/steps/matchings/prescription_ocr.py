from pills_identification.workflows import IWorkflowStep
from typing import List
from unidecode import unidecode
from paddleocr import PaddleOCR
import numpy as np
import json
import torchmetrics
import itertools
import os
import argparse
from pathlib import Path



class PrescriptionOCR(IWorkflowStep):
    def __call__(self, images: List[np.array], **kwargs) -> List[int]:
        """get text from OCR image
        
        Args:
        images List[np.array]: List of prescription images
        Returns:
            List[int]:  drugids
        """
        ocr_model = PaddleOCR(lang='en', show_log = False, debug = False, use_angle_cls=True)
        all_objects = [ocr_model.ocr(image) for image in images]

        drugname_objects= [[re for re in obj if re[1][0].split(')')[0].strip().isnumeric() and ')' in re[1][0]] for obj in all_objects]
        drugnames = [[unidecode(item[1][0][item[1][0].find(')')+1:]).lower().replace(" ", "") for item in obj] for obj in drugname_objects]

        script_dir = os.path.dirname(__file__) 
        rel_path = "drug_id_mapping.json"
        abs_file_path = os.path.join(script_dir, rel_path)       
        with open(abs_file_path) as f:
           mapping = json.load(f)
        
        drugids = []
        metric = torchmetrics.CharErrorRate()
        groundtruth_drugname = mapping.keys()
        for drugname in drugnames:
            for item in drugname:
                minError = 1
                drugname_map = ''
                for drugname in groundtruth_drugname:
                    if metric(item, drugname) < minError:
                        minError = metric(item, drugname)
                        drugname_map = drugname
                drugids.append(mapping[drugname_map])
               
        drugids = list(itertools.chain(*drugids))

        return {**kwargs, "drug_ids": drugids}