import cv2
import warnings
from src.ocr_engine import OcrEngine
import enchant
from utils.preprocess_utils import draw_boxes

warnings.filterwarnings("ignore", category=FutureWarning)
MOUSE_X, MOUSE_Y = None, None


def check_word(word: str, language: str = "es"):
    dictionary = enchant.Dict(language)
    return dictionary.check(word)


def adjust_bounding_boxes(results, start_x, start_y):
    for result in results:
        bbox = result["bbox"]
        adjusted_bbox = [[x + start_x, y + start_y] for x, y in bbox]
        result["bbox"] = adjusted_bbox
    return results


def handle_frame(frame, ocr):
    global MOUSE_X, MOUSE_Y
    width, height = int(frame.shape[1] * 0.9), int((frame.shape[1] * 0.9) // 2)
    results = None
    word_finded = False
    final_word, final_results = None, None
    decrease_factor = 0.05  # Increase size by 5% each iteration

    while True:
        cv2.setMouseCallback("Frame", click_event)
        if word_finded:
            # Draw adjusted bounding boxes on the original frame
            cv2.imshow(
                "Frame",
                draw_boxes(image=frame, detections=results),
            )
        else:
            cv2.imshow("Frame", frame)

        if MOUSE_X is not None and MOUSE_Y is not None:
            while not word_finded:
                # Ensure cropping does not go out of bounds
                start_x = max(0, MOUSE_X - width // 2)
                start_y = max(0, MOUSE_Y - height // 2)
                end_x = min(frame.shape[1], start_x + width)
                end_y = min(frame.shape[0], start_y + height)

                # Crop the frame
                crop_frame = frame[start_y:end_y, start_x:end_x]
                results = ocr.inference(crop_frame)

                if len(results) == 1:
                    words = results[0]["text"]
                    final_word = words.replace(",", " ").split()
                    if len(final_word) == 1:
                        print(final_word[0])
                        check_word(final_word[0])
                        # Adjust bounding boxes to original frame
                        results = adjust_bounding_boxes(
                            results, start_x, start_y
                        )
                        final_results = results
                        word_finded = True

                # Increase size proportionally
                width = int(width * (1 - decrease_factor))
                height = int(height * (1 - decrease_factor))

                # Ensure size does not become too small
                if width <= 0 or height <= 0:
                    break

                print(height, width)

        if cv2.waitKey(1) == ord("q"):
            MOUSE_X, MOUSE_Y = None, None
            break


def click_event(event, x, y, flags, params):
    global MOUSE_X, MOUSE_Y
    # Check if left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: X = {x}, Y = {y}")
        MOUSE_X, MOUSE_Y = x, y


def start_video():
    cap = cv2.VideoCapture(0)
    ocr = OcrEngine()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Key for action
        key = cv2.waitKey(1)

        # Character "a" for detect words
        if key == ord("a"):
            handle_frame(frame, ocr)
        elif key == ord("q"):
            break

        cv2.imshow("Frame", frame)

    cap.release()
    cv2.destroyAllWindows()


def main():
    start_video()


if __name__ == "__main__":
    main()
