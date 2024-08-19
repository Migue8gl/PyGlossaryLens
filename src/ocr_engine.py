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

    def _inference_easyocr(
        self, data: Union[np.ndarray, str, List[str]], data_detail: int = 1
    ) -> List[Any]:
        # Check if data is a list and not empty
        if isinstance(data, list) and data:
            # Process each image in the list and return the OCR results
            inferences = [
                self.ocr.readtext(img, detail=data_detail) for img in data
            ]
            return inferences
        elif isinstance(data, str):
            # If data is a string and it's a file path, read the image
            if os.path.isfile(data):
                image = cv2.imread(data)
                return self.ocr.readtext(image, detail=data_detail)
            else:
                # If the file doesn't exist, raise an exception
                raise FileNotFoundError(f"The file {data} does not exist.")
        elif isinstance(data, np.ndarray):
            # If data is a numpy array (image), process it directly
            return self.ocr.readtext(data, detail=data_detail)
        else:
            # If data type is unsupported, raise an exception
            raise TypeError("Unsupported data type for inference.")

    def _inference_mmocr(
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

    def _parse_output(self, ocr_output: Any) -> List[dict]:
        parsed_results = []

        # Parse distinct ocrs
        if self.engine == "easyocr":
            for detection in ocr_output:
                # EasyOCR output format: [bounding_box, text, confidence]
                bounding_box, text, confidence = detection
                parsed_results.append(
                    {
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [
                            [int(coord) for coord in point]
                            for point in bounding_box
                        ],
                    }
                )
        return parsed_results

    def inference(self, data: Union[np.ndarray, str, List[str]]) -> List[Any]:
        if self.engine == "easyocr":
            return self._parse_output(self._inference_easyocr(data))
        else:
            return self._inference_mmocr(data)
