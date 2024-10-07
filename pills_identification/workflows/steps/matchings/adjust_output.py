from typing import List

from pills_identification.workflows import IWorkflowStep


class AdjustOutputStep(IWorkflowStep):
    def __call__(self, pill_ids: List[int], prescription_ids: List[int], **kwargs) -> List[int]:
        """_summary_
        Args:
            pill_ids List[int]: List of drugIds from classification model
            prescription_ids List[int]: List drugIds from prescription (OCR model)
        Raises:

        Returns:
            List[int]: List of drugIds appear in both the prescription and pill images.
        """

        pill_paths = kwargs["pill_paths"]
        pill_ids = [int(i) for i in pill_ids]

        match_ids = []

        for i, id_ in enumerate(pill_ids):
            pill_name = pill_paths[i].split("/")[-1].split(".")[0]
            pill_name = "_".join(pill_name.split("_")[:-1])

            if id_ in prescription_ids[pill_name]:
                match_ids.append(id_)
            else:
                match_ids.append(107)

        return {**kwargs, "match_ids": match_ids, "prescription_ids": prescription_ids, "pill_ids": pill_ids}
