import glob
import re
import cv2
import random
from pprint import pprint
import json
import argparse


from pills_identification.workflows.pills_workflow import PillsWorkflow
from pills_identification.workflows.steps.image_io.opencv import OpenCVImageReader
from pills_identification.workflows.steps.localizations.yolov5 import YOLOv5LocalizationStep
from pills_identification.workflows.steps.alignments.remove_background import DeepRemoveBackgroundStep
from pills_identification.workflows.steps.alignments.rotate_pill import RotatePillStep
from pills_identification.workflows.steps.classifiers.EfficientNetV2S import EfficientNetV2SClassificationStep
from pills_identification.workflows.steps.matchings.adjust_output import AdjustOutputStep

from pills_identification.tools.reduce_boundingbox import ReduceBoundingBox
from pills_identification.tools.generate_submission import GenerateSubmission



# from pills_identification.workflows.steps.matchings.prescription_tesseract_ocr import PrescriptionTesseractOCR
from pills_identification.workflows.steps.matchings.prescription_paddle_ocr import PrescriptionPaddleOCR


def main(image_paths, output_path, json_ocr, batch_size):

    if image_paths[-1] != "/":
        image_paths += "/"

    test_img_path = image_paths + "*.jpg"
    pill_file_paths = sorted(list(glob.glob(test_img_path)))[:100]


    batches = [pill_file_paths[i : i + batch_size] for i in range(0, len(pill_file_paths), batch_size)]


    pill_steps = [
        OpenCVImageReader(),
        YOLOv5LocalizationStep(model_path="/workspace/Dev/hc-labs/aiml-research/pills-identification/weights/yolov5.pt"),
        # DeepRemoveBackgroundStep(),
        # RotatePillStep(),
        # ReduceBoundingBox(),
        EfficientNetV2SClassificationStep(model_path="/workspace/Dev/hc-labs/aiml-research/pills-identification/weights/EfficientNetV2S.h5")
    ]

    pill_workflow = PillsWorkflow(steps=pill_steps)    
    pill_results = [pill_workflow(file_paths=pill_file_path) for pill_file_path in batches]

    final_step = [AdjustOutputStep(), GenerateSubmission()]

    final_workflow = PillsWorkflow(steps=final_step)

    with open(json_ocr, 'r') as f:
        pres_results = json.load(f)

    results = [final_workflow(**pill_result, **pres_results) for pill_result in pill_results]

    result_txt = "image_name,class_id,confidence_score,x_min,y_min,x_max,y_max\n"

    for item in results:
        result_txt += item


    with open(output_path, "w") as f:
        f.write(result_txt)

    return result_txt

if __name__ == "__main__":

         # python examples/submission_workflow.py -i /workspace/Research/Pill/vaipe/public_test/pill/image -o submit.txt -j ocr_1.json -b 8
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input_images", help="Path of inputs image")
        parser.add_argument("-o", "--output_path", help="Path of output submission")
        parser.add_argument("-j", "--json_ocr", help="Path of OCR result")
        parser.add_argument("-b", "--batch_size", help="Batch size", default=8)

        args = parser.parse_args()


        input_paths = args.input_images
        output_paths = args.output_path
        json_ocr = args.json_ocr
        batch_size = int(args.batch_size)

        print("INPUT ", input_paths)

        result = main(input_paths, output_paths, json_ocr, batch_size)

        print(result)


