from typing import Union, List, Optional, Any
import numpy as np
import os
import cv2


class OcrEngine:
    def __init__(
        self, language: Optional[str] = None, engine="easyocr"
    ) -> None:
        self.lang = language if language is not None else "es"
        self.engine = engine
        if self.engine == "easyocr":
            from easyocr import Reader

            self.ocr = Reader([self.lang], gpu=False, verbose=False)

    def inference_easyocr(
        self, data: Union[np.ndarray, str, List[str]]
    ) -> List[Any]:
        # Check if data is a list and not empty
        if isinstance(data, list) and data:
            # Process each image in the list and return the OCR results
            inferences = [self.ocr.readtext(img) for img in data]
            return inferences
        elif isinstance(data, str):
            # If data is a string and it's a file path, read the image
            if os.path.isfile(data):
                image = cv2.imread(data)
                return self.ocr.readtext(image)
            else:
                # If the file doesn't exist, raise an exception
                raise FileNotFoundError(f"The file {data} does not exist.")
        elif isinstance(data, np.ndarray):
            # If data is a numpy array (image), process it directly
            return self.ocr.readtext(data)
        else:
            # If data type is unsupported, raise an exception
            raise TypeError("Unsupported data type for inference.")

    def inference_mmocr(
        self, data: Union[np.ndarray, str, List[str]]
    ) -> List[Any]:
        # Check if data is a list and not empty
        if isinstance(data, list) and data:
            # Process each image in the list and return the OCR results
            inferences = [self.ocr(img) for img in data]
            return inferences
        elif isinstance(data, str):
            # If data is a string and it's a file path, read the image
            if os.path.isfile(data):
                return self.ocr(data)
            else:
                # If the file doesn't exist, raise an exception
                raise FileNotFoundError(f"The file {data} does not exist.")
        elif isinstance(data, np.ndarray):
            # If data is a numpy array (image), process it directly
            return self.ocr(data)
        else:
            # If data type is unsupported, raise an exception
            raise TypeError("Unsupported data type for inference.")

    def inference(self, data: Union[np.ndarray, str, List[str]]) -> List[Any]:
        if self.engine == "easyocr":
            return self.inference_easyocr(data)
        else:
            return self.inference_mmocr(data)
