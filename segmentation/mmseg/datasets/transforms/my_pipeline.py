from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@TRANSFORMS.register_module()
class ConvertTo8Bit:
    """Convert images to 8-bit depth."""

    def __call__(self, results):
        img = results['img']
        if img.dtype == np.int16 or img.dtype == np.uint16:
            # Ensure correct scaling when converting from 16-bit to 8-bit
            img = (img / 256).astype(np.uint8)
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class ConvertToGrayScaleMask(BaseTransform):
    """Converts a segmentation map to a binary mask rather than the color coded one.

    Required Keys:
    - gt_seg_map

    Modified Keys:
    - gt_seg_map

    Args:
        None
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: dict) -> dict:
        """Call function to convert seg map to binary mask.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Converted results.
        """

        for key in results.get("seg_fields", []):
            if len(results[key].shape) == 3 and results[key].shape[2] == 3:
                results[key] = cv2.cvtColor(results[key], cv2.COLOR_BGR2GRAY)

            results[key] = np.where(results[key] > 0, 1, 0)
            # if np.any(results[key] == 1):
            #    print("There is at least one 1 in the array")
            # else:
            #    print("No 1s exist.")
        return results
