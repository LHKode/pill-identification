import cv2
import numpy as np
from typing import List
from math import atan2

from pills_identification.workflows import IWorkflowStep


class RotatePillStep(IWorkflowStep):

    def get_contours(self, img: np.array):
        gray = cv2.cvtColor(cv2.bitwise_and(img,img,mask=img[:,:,3]),cv2.COLOR_BGRA2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        cnts,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final = cnts[0]
        for c in cnts:
            if len(final) == 0 or len(c)>len(final):
                final = c
        contour = np.array(final).astype(np.int32)
        return contour

    def get_orientation(self, contours, image: np.array) -> int:
        # Construct a buffer used by the pca analysis
        size = len(contours)
        data_contours = np.empty((size, 2), dtype=np.float64)
        for i in range(data_contours.shape[0]):
            data_contours[i,0] = contours[i,0,0]
            data_contours[i,1] = contours[i,0,1]
        
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, _ = cv2.PCACompute2(data_contours, mean)
        
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        
        return int(np.rad2deg(angle))

    def __call__(self, images: List[np.array], **kwargs) -> List[np.array]:
        """_summary_
        Args:
            images List[np.array]: List of images in RGB color mode
        Raises:
            
        Returns:
            List[np.array]: List of images in RGB color mode
        """
        extend_ratio = 0.5
        padding_pills = [cv2.copyMakeBorder(image,
                            int(extend_ratio * image.shape[0]), 
                            int(extend_ratio * image.shape[0]), 
                            int(extend_ratio * image.shape[1]),
                            int(extend_ratio * image.shape[1]), 
                            cv2.BORDER_REPLICATE,
                            None, (255,255,255)) for image in images]

        angles = [self.get_orientation(self.get_contours(image), image)  for image in padding_pills]

        rotated_images = []
        for padding_pill, angle in list(zip(padding_pills, angles)):
            rows, cols, _ = padding_pill.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            rotated_images.append(cv2.warpAffine(padding_pill,rotation_matrix,(cols,rows)))

        return {**kwargs, "images": rotated_images}