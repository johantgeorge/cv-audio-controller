# python3 -m venv .venv
# source .venv/bin/activate
# pip install opencv-python ollama

import cv2
import time
import tempfile
import threading
from ollama import chat

MODEL = "moondream:1.8b"

# Seconds between model calls.
# Lower = faster updates but more lag.
# Higher = smoother camera but slower classification updates.
ANALYZE_TIME = 0.25

FRAME_SIZE = (224, 224)

NO_GESTURE = "No Gesture Detected"

GESTURES = [
    "Wave",
    "Fist",
    "Peace Sign",
    "Thumbs Up",
    "Pointer Finger",
    "Middle Finger",
    "Ring Finger",
    
    NO_GESTURE,
]

gesture_options = ", ".join(GESTURES)

description = "Looking..."
is_processing = False
lock = threading.Lock()

SYSTEM_PROMPT = (
    "You are a hand gesture classifier. "
    f"Choose exactly one label from this list: {gesture_options}. "
    f"If no hand is visible, or if the gesture does not clearly match one of the listed labels, return {NO_GESTURE}. "
    "Return only the label and nothing else."
)

BASE_MESSAGES = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]


def normalize_label(raw_text):
    """
    Converts model output into one valid gesture label.
    Defaults to NO_GESTURE if the output is unclear.
    """
    raw = raw_text.strip().lower()

    # Exact match first
    for gesture in GESTURES:
        if raw == gesture.lower():
            return gesture

    # Fallback partial match
    for gesture in GESTURES:
        if gesture.lower() in raw:
            return gesture

    return NO_GESTURE


def describe_image_async(frame):
    global description, is_processing

    try:
        small_frame = cv2.resize(frame, FRAME_SIZE)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp:
            success = cv2.imwrite(temp.name, small_frame)

            if not success:
                raise RuntimeError("Failed to write temporary image.")

            response = chat(
                model=MODEL,
                messages=[
                    *BASE_MESSAGES,
                    {
                        "role": "user",
                        "content": "Classify the hand gesture in this frame.",
                        "images": [temp.name],
                    },
                ],
            )

        raw_response = response["message"]["content"]
        label = normalize_label(raw_response)

        with lock:
            description = label

    except Exception as e:
        with lock:
            description = f"Error: {e}"

    finally:
        with lock:
            is_processing = False


def main():
    global is_processing

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open MacBook camera.")
        return

    # Optional: reduce camera capture resolution for smoother preview
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press q or Esc in the camera window to quit. Ctrl+C also works.")

    last_describe_time = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            current_time = time.time()

            with lock:
                current_description = description
                currently_processing = is_processing

            should_analyze = (
                current_time - last_describe_time >= ANALYZE_TIME
                and not currently_processing
            )

            if should_analyze:
                last_describe_time = current_time

                with lock:
                    is_processing = True

                frame_copy = frame.copy()

                thread = threading.Thread(
                    target=describe_image_async,
                    args=(frame_copy,),
                    daemon=True,
                )
                thread.start()

            cv2.putText(
                frame,
                current_description[:90],
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Live Camera Vision", frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()