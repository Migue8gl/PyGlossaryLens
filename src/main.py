import cv2
from deskew import determine_skew
from easyocr import Reader
from utils.preprocess_frame import draw_boxes, rotate, binarize
from utils.geometry_utils import compute_rectangle_given_center
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
MOUSE_X, MOUSE_Y = None, None


def extract_text(data: dict[str, str]):
    text = data["text"]
    text_list = list(map(lambda x: " " if x == "" else x, text))
    return "".join(text_list)


def show_frame(frame):
    width, height = 0.2, 0.1
    while True:
        cv2.imshow("Frame", frame)
        cv2.setMouseCallback("Frame", click_event)

        if MOUSE_X is not None and MOUSE_Y is not None:
            vertices = compute_rectangle_given_center(
                MOUSE_X, MOUSE_Y, width, height
            )

        if cv2.waitKey(1) == ord("q"):
            break


def click_event(event, x, y, flags, params):
    global MOUSE_X, MOUSE_Y
    # Check if left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: X = {x}, Y = {y}")
        MOUSE_X, MOUSE_Y = x, y


def start_video():
    cap = cv2.VideoCapture(0)
    reader = Reader(["es"], gpu=False, verbose=False)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Frame with boxes surrounding words detected
        frame_with_boxes = frame

        # Key for action
        key = cv2.waitKey(1)

        # Character "a" for detect words
        if key == ord("a"):
            angle = determine_skew(frame)
            transformed_frame = binarize(frame)
            transformed_frame = rotate(transformed_frame, angle, (0, 0, 0))

            # Use EasyOCR to extract text
            results = reader.readtext(frame)

            if results and isinstance(results, list):
                text = ""

                for result in results:
                    # Concatenate text with space
                    if len(result) > 0:
                        text += result[1] + " "
                with open("files/extracted_text.txt", "w") as file:
                    file.write(text.strip())

                # Draw bounding boxes on the frame and correct angle
                frame_with_boxes = draw_boxes(frame, results, angle)

                show_frame(frame_with_boxes)
        elif key == ord("q"):
            break

        cv2.imshow("Frame", frame_with_boxes)

    cap.release()
    cv2.destroyAllWindows()


def main():
    start_video()


if __name__ == "__main__":
    main()
