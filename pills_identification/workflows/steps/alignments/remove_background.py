from typing import Dict, List
import numpy as np
from rembg import remove
from tqdm import tqdm

from pills_identification.workflows import IWorkflowStep


class RemoveBackgroundStep(IWorkflowStep):
    def __call__(self, images: List[np.array], metadata: List[Dict], **kwargs):
        """__call__ 

        Parameters
        ----------
        images : List[np.array]
            List of input images
        metadata : List[Dict]
            List of metadata coresponding to images

        Raises
        ------
        NotImplementedError
            This step was not implemented
        """
        print(RemoveBackgroundStep.__name__ + "receive" + kwargs)
        raise NotImplementedError("RemoveBackgroundStep was not implemented")


class DeepRemoveBackgroundStep(IWorkflowStep):
    def __call__(self, images: List[np.array], **kwargs) -> Dict[str, List[np.array]]:
        """Remove background of pill images

        Args:
            images List[np.array]]: List of pill images

        Returns:
            Dict[str, List[np.array]]: Pill images without background
        """
        no_bg_images = []
        for image in tqdm(images):
            no_bg_images.append(remove(image, alpha_matting=True))

        return {**kwargs, "images": no_bg_images}
