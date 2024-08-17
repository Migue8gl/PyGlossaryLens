import math
from typing import Tuple, Union

import cv2
import easyocr as eo
import numpy as np
from deskew import determine_skew


def rotate(
    image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width
    )
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height
    )
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(
        image, rot_mat, (int(round(height)), int(round(width))), borderValue=background
    )


def rotated_rectangle(image, start_point, end_point, color, thickness, rotation=0):
    """
    https://stackoverflow.com/questions/68423495/how-do-i-change-the-angle-of-a-cv2-rectangle
    """
    center_point = [
        (start_point[0] + end_point[0]) // 2,
        (start_point[1] + end_point[1]) // 2,
    ]
    height = end_point[1] - start_point[1]
    width = end_point[0] - start_point[0]
    angle = np.radians(rotation)

    # Determine the coordinates of the 4 corner points
    rotated_rect_points = []
    x = center_point[0] + ((width / 2) * np.cos(angle)) - ((height / 2) * np.sin(angle))
    y = center_point[1] + ((width / 2) * np.sin(angle)) + ((height / 2) * np.cos(angle))
    rotated_rect_points.append([x, y])
    x = center_point[0] - ((width / 2) * np.cos(angle)) - ((height / 2) * np.sin(angle))
    y = center_point[1] - ((width / 2) * np.sin(angle)) + ((height / 2) * np.cos(angle))
    rotated_rect_points.append([x, y])
    x = center_point[0] - ((width / 2) * np.cos(angle)) + ((height / 2) * np.sin(angle))
    y = center_point[1] - ((width / 2) * np.sin(angle)) - ((height / 2) * np.cos(angle))
    rotated_rect_points.append([x, y])
    x = center_point[0] + ((width / 2) * np.cos(angle)) + ((height / 2) * np.sin(angle))
    y = center_point[1] + ((width / 2) * np.sin(angle)) - ((height / 2) * np.cos(angle))
    rotated_rect_points.append([x, y])
    cv2.polylines(
        image, np.array([rotated_rect_points], np.int32), True, color, thickness
    )


def draw_boxes(image: np.ndarray, detections, angle: float, threshold=0.25):
    for bbox, _, score in detections:
        if score > threshold:
            rotated_rectangle(
                image,
                tuple(map(int, bbox[0])),
                tuple(map(int, bbox[2])),
                (0, 255, 0),
                2,
                angle,
            )
    return image


def binarize(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.adaptiveThreshold(
        grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5
    )
    return threshold_img


def noise_removal(image):
    import numpy as np

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def remove_borders(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y : y + h, x : x + w]
    return crop


def extract_text(data: dict[str, str]):
    text = data["text"]
    text_list = list(map(lambda x: " " if x == "" else x, text))
    return "".join(text_list)


def start_video():
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))
    reader = eo.Reader(["es"], gpu=False, verbose=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        """
        transformed_frame = binarize(frame)
        transformed_frame = remove_borders(transformed_frame)
        angle = determine_skew(transformed_frame)
        transformed_frame = rotate(transformed_frame, angle, (0, 0, 0))
        """
        if cv2.waitKey(1) == ord("a"):
            angle = determine_skew(frame)
            # Use EasyOCR to extract text
            results = reader.readtext(frame)

            # Print results to debug and understand the structure
            print("OCR Results:", results)

            if results and isinstance(results, list):
                text = ""
                for result in results:
                    if len(result) > 1:
                        text += result[1] + " "  # Concatenate text with space
                # Write text to file
                with open("extracted_text.txt", "w") as file:
                    file.write(text.strip())

                # Draw bounding boxes on the frame
                frame_with_boxes = draw_boxes(frame, results, angle)
            else:
                # No results or unexpected structure
                frame_with_boxes = frame
        else:
            frame_with_boxes = frame

        cv2.imshow("Frame", frame_with_boxes)
        out.write(frame_with_boxes)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    start_video()


if __name__ == "__main__":
    main()
