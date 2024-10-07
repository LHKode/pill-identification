import argparse
from typing import List
from pills_identification.workflows.pills_workflow import PillsWorkflow, PillsWorkflowStep


def bb2str(bb):
    return ",".join([str(int(i)) for i in bb])


class GenerateSubmission(PillsWorkflowStep):
    def __call__(
        self, pill_paths: List[str], match_ids: List[int], confidence_scores: List[float], bounding_boxes: List[List[float]], **kwargs
    ) -> str:
        result_length = len(pill_paths)        
        submission_txt = ""

        bounding_boxes = [bb2str(i) for i in bounding_boxes]

        for i in range(result_length):
            submission_txt += \
                f"{pill_paths[i].split('/')[-1]}," \
                +f"{match_ids[i]}," \
                +f"{confidence_scores[i]:.2f}," \
                +f"{bounding_boxes[i]}\n"

        return submission_txt


if __name__ == "__main__":
    pass