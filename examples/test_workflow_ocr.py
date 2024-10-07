import glob

from pills_identification.workflows.pills_workflow import PillsWorkflow
from pills_identification.workflows.steps.image_io.opencv import OpenCVImageReader
from pills_identification.workflows.steps.matchings.prescription_ocr import PrescriptionOCR


def main():
    steps = [
        OpenCVImageReader(),
        PrescriptionOCR(),
    ]
    file_paths = list(glob.glob(f"/home/khanhlam/Desktop/HC-lab/VAIPE_P_TEST_0.png"))
    print(file_paths)
    workflow = PillsWorkflow(steps=steps)
    results = workflow(file_paths=file_paths)
    print(results)

if __name__ == "__main__":
    main()