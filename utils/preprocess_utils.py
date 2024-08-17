import math
from typing import Tuple, Union
import numpy as np
import cv2


def rotate(
    image: np.ndarray,
    angle: float,
    background: Union[int, Tuple[int, int, int]] = (0, 0, 0),
) -> np.ndarray:
    """
    Rotate an image by a given angle.

    Args:
        image (np.ndarray): Input image to rotate.
        angle (float): Angle of rotation in degrees.
        background (Union[int, Tuple[int, int, int]]): Background color for new pixels. Default is black.

    Returns:
        np.ndarray: Rotated image.
    """
    old_height, old_width = image.shape[:2]
    angle_radian = math.radians(angle)
    width = int(
        abs(np.sin(angle_radian) * old_height)
        + abs(np.cos(angle_radian) * old_width)
    )
    height = int(
        abs(np.sin(angle_radian) * old_width)
        + abs(np.cos(angle_radian) * old_height)
    )

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2

    return cv2.warpAffine(
        image, rot_mat, (width, height), borderValue=background
    )


def rotate_point(x, y, w, h, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Center of the image
    cx, cy = w / 2, h / 2

    # Translate point to origin (center of image)
    x_translated = x - cx
    y_translated = y - cy

    # Apply rotation
    x_rotated = x_translated * math.cos(
        angle_radians
    ) - y_translated * math.sin(angle_radians)
    y_rotated = x_translated * math.sin(
        angle_radians
    ) + y_translated * math.cos(angle_radians)

    # Translate point back
    x_new = x_rotated + cx
    y_new = y_rotated + cy

    return int(x_new), int(y_new)


def binarize(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to binary (black and white) using adaptive thresholding.

    Args:
        image (np.ndarray): Input image to binarize.

    Returns:
        np.ndarray: Binarized image.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )


def noise_removal(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from a binary image using morphological operations.

    Args:
        image (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Noise-removed image.
    """
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.medianBlur(image, 3)


def remove_borders(image: np.ndarray) -> np.ndarray:
    """
    Remove borders from a binary image by finding the largest contour.

    Args:
        image (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Image with borders removed.
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y : y + h, x : x + w]


def rotated_rectangle(
    image: np.ndarray,
    start_point: Tuple[int, ...],
    end_point: Tuple[int, ...],
    color: Tuple[int, int, int],
    thickness: int,
    rotation: float = 0,
) -> None:
    """
    Draw a rotated rectangle on an image.

    Args:
        image (np.ndarray): Input image to draw on.
        start_point (Tuple[int, int]): Top-left corner of the rectangle before rotation.
        end_point (Tuple[int, int]): Bottom-right corner of the rectangle before rotation.
        color (Tuple[int, int, int]): Color of the rectangle in BGR format.
        thickness (int): Thickness of the rectangle lines.
        rotation (float): Rotation angle in degrees. Default is 0.
    """
    # Compute the center, width, and height of the rectangle
    center_point = (
        (start_point[0] + end_point[0]) // 2,
        (start_point[1] + end_point[1]) // 2,
    )
    width = end_point[0] - start_point[0]
    height = end_point[1] - start_point[1]

    # Convert rotation angle from degrees to radians
    angle_rad = np.radians(rotation)

    # Define the rectangle's four corners before rotation
    rect_points = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )

    # Rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    # Rotate all four points
    rotated_rect_points = np.dot(rect_points, rotation_matrix.T)

    # Shift points back to the center
    rotated_rect_points += np.array(center_point)

    # Convert to integer and reshape for polylines
    rotated_rect_points = rotated_rect_points.astype(int).reshape((-1, 1, 2))

    # Draw the rotated rectangle on the image
    cv2.polylines(image, [rotated_rect_points], True, color, thickness)


def draw_boxes(
    image: np.ndarray, detections: list, angle: float, threshold: float = 0.1
) -> np.ndarray:
    """
    Draw bounding boxes on an image based on detections.

    Args:
        image (np.ndarray): Input image to draw on.
        detections (list): List of detections, each containing bounding box coordinates and score.
        angle (float): Rotation angle for the rectangles.
        threshold (float): Confidence threshold for drawing boxes. Default is 0.1.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
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
