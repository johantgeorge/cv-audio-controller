import time
from collections import Counter, deque

import cv2
import mediapipe as mp

import os
import subprocess

# Spotify -> Spotify
# Music   -> Apple Music
MUSIC_APP = "Spotify"

MODEL_PATH = "gesture_recognizer.task"

NO_GESTURE = "No Gesture Detected"

PREDICTION_BUFFER_SIZE = 7
WAVE_BUFFER_SIZE = 20

INTRA_HAND_COLOR = (255, 255, 255)   # white
INTER_HAND_COLOR = (0, 255, 0)       # green

WAVE_MIN_X_RANGE = 0.18
WAVE_MIN_DIRECTION_CHANGES = 1

MIN_GESTURE_SCORE = 0.45

MAX_HANDS = 2

# Swap this if Left/Right appear backwards because of the mirrored camera view.
SWAP_HANDEDNESS = True

# -----------------------------
# macOS Volume 
# -----------------------------

VOLUME_STEP = 4


CONTROL_COOLDOWN = 0.12
VOLUME_CONTROL_COOLDOWN = 0.08
TRACK_SKIP_COOLDOWN = 1.5

MIN_VOLUME = 0
MAX_VOLUME = 100



BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def map_mediapipe_label(label: str, score: float) -> str:
    """
    Convert MediaPipe's raw labels into nicer display names.
    """
    if score < MIN_GESTURE_SCORE:
        return "No Gesture"

    display_names = {
        "None": "No Gesture",
        "Closed_Fist": "Fist",
        "Open_Palm": "Open Palm",
        "Pointing_Up": "Pointer",
        "Thumb_Down": "Thumbs Down",
        "Thumb_Up": "Thumbs Up",
        "Victory": "Peace Sign",
        "ILoveYou": "Rockstar",
    }

    return display_names.get(label, label)


def normalize_handedness(hand_name: str | None) -> str | None:
    """
    Optionally swap Left/Right if your mirrored camera feed makes them appear reversed.
    """
    if hand_name not in ("Left", "Right"):
        return None

    if not SWAP_HANDEDNESS:
        return hand_name

    if hand_name == "Left":
        return "Right"

    return "Left"


def get_handedness(result, hand_idx: int) -> str | None:
    """
    Return 'Left' or 'Right' for the detected hand at hand_idx.
    """
    if not result.handedness:
        return None

    if hand_idx >= len(result.handedness):
        return None

    if not result.handedness[hand_idx]:
        return None

    handedness_category = result.handedness[hand_idx][0]
    hand_name = handedness_category.category_name

    return normalize_handedness(hand_name)


def get_static_gesture_for_hand(result, hand_idx: int) -> tuple[str, float]:
    """
    Return the top static gesture and score for one detected hand.
    """
    if not result.gestures:
        return NO_GESTURE, 0.0

    if hand_idx >= len(result.gestures):
        return NO_GESTURE, 0.0

    if not result.gestures[hand_idx]:
        return NO_GESTURE, 0.0

    top_category = result.gestures[hand_idx][0]
    label = top_category.category_name
    score = top_category.score

    return map_mediapipe_label(label, score), score


def get_hand_center_x(landmarks) -> float:
    """
    Get normalized x-center of one detected hand.
    """
    xs = [landmark.x for landmark in landmarks]
    return sum(xs) / len(xs)


def is_wave_motion(x_positions: deque) -> bool:
    """
    Detect a simple left-right wave using recent hand x positions.
    """
    if len(x_positions) < WAVE_BUFFER_SIZE:
        return False

    xs = list(x_positions)

    x_range = max(xs) - min(xs)
    if x_range < WAVE_MIN_X_RANGE:
        return False

    deltas = []

    for i in range(1, len(xs)):
        delta = xs[i] - xs[i - 1]

        # Ignore tiny movement/noise.
        if abs(delta) > 0.01:
            deltas.append(delta)

    if len(deltas) < 4:
        return False

    signs = [1 if delta > 0 else -1 for delta in deltas]

    direction_changes = 0

    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            direction_changes += 1

    return direction_changes >= WAVE_MIN_DIRECTION_CHANGES


def smooth_prediction(predictions: deque) -> str:
    """
    Majority vote over recent predictions to reduce flicker.
    """
    if not predictions:
        return NO_GESTURE

    counts = Counter(predictions)
    return counts.most_common(1)[0][0]


def draw_landmarks(frame, result):
    """
    Draw hand landmarks manually.

    This avoids using mp.solutions, which is not exposed in your installed
    mediapipe package.
    """
    if not result.hand_landmarks:
        return

    height, width, _ = frame.shape

    connections = [
        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index finger
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle finger
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring finger
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    for hand_landmarks in result.hand_landmarks:
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(
                    frame,
                    points[start_idx],
                    points[end_idx],
                    #(255, 255, 255), # color
                    INTRA_HAND_COLOR,
                    2,
                )

        for x, y in points:
            cv2.circle(
                frame,
                (x, y),
                4,
                #(255, 255, 255),
                INTRA_HAND_COLOR,
                -1,
            )

def draw_inter_hand_connections(frame, result):
    """
    Draw colored connections between the two hands.
    Connects corresponding landmark points across both hands.
    """
    if not result.hand_landmarks or len(result.hand_landmarks) < 2:
        return

    height, width, _ = frame.shape

    hand_a = result.hand_landmarks[0]
    hand_b = result.hand_landmarks[1]

    points_a = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_a
    ]
    points_b = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_b
    ]

    INTER_HAND_LANDMARK_INDICES = [0, 4, 8, 12, 16, 20]

    for idx in INTER_HAND_LANDMARK_INDICES:
        if idx < len(points_a) and idx < len(points_b):
            cv2.line(
                frame,
                points_a[idx],
                points_b[idx],
                INTER_HAND_COLOR,
                2,
            )

            cv2.circle(frame, points_a[idx], 5, INTER_HAND_COLOR, -1)
            cv2.circle(frame, points_b[idx], 5, INTER_HAND_COLOR, -1)

def draw_hand_labels_near_hands(frame, result, current_labels, current_scores):
    """
    Draw Left/Right labels near each detected hand.
    """
    if not result.hand_landmarks:
        return

    height, width, _ = frame.shape

    for hand_idx, landmarks in enumerate(result.hand_landmarks):
        hand_name = get_handedness(result, hand_idx)

        if hand_name not in ("Left", "Right"):
            continue

        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]

        min_x = int(min(xs) * width)
        min_y = int(min(ys) * height)

        label = current_labels[hand_name]
        score = current_scores[hand_name]

        text = f"{hand_name}: {label} ({score:.2f})"

        cv2.putText(
            frame,
            text,
            (min_x, max(min_y - 15, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

def get_current_volume():
    """
    Get current macOS output volume.
    """
    try:
        result = subprocess.check_output(
            [
                "osascript",
                "-e",
                "output volume of (get volume settings)"
            ]
        )

        return int(result.decode().strip())

    except Exception:
        return 50


def set_volume(volume):
    """
    Set macOS system volume.
    """
    volume = max(MIN_VOLUME, min(MAX_VOLUME, int(volume)))

    os.system(
        f"osascript -e 'set volume output volume {volume}'"
    )

    return volume


def play_track():
    os.system(f"""
    osascript -e 'tell application "{MUSIC_APP}" to play' 
    """)

def pause_track():
    os.system(f"""
    osascript -e 'tell application "{MUSIC_APP}" to pause'
    """)

def prev_track():
    os.system(f"""
    osascript -e 'tell application "{MUSIC_APP}" to previous track'
    """)

def next_track():
    os.system(f"""
    osascript -e 'tell application "{MUSIC_APP}" to next track'
    """)





def main():
    prediction_buffers = {
        "Left": deque(maxlen=PREDICTION_BUFFER_SIZE),
        "Right": deque(maxlen=PREDICTION_BUFFER_SIZE),
    }

    wave_x_buffers = {
        "Left": deque(maxlen=WAVE_BUFFER_SIZE),
        "Right": deque(maxlen=WAVE_BUFFER_SIZE),
    }

    current_labels = {
        "Left": NO_GESTURE,
        "Right": NO_GESTURE,
    }

    current_scores = {
        "Left": 0.0,
        "Right": 0.0,
    }

    fps_buffer = deque(maxlen=30)

    # -----------------------------------
    # macOS control state
    # -----------------------------------

    VOLUME_STEP = 3

    CONTROL_COOLDOWN = 0.12
    VOLUME_CONTROL_COOLDOWN = 0.08
    TRACK_SKIP_COOLDOWN = 1.5

    last_control_time = 0
    last_track_skip_time = 0

    current_volume = get_current_volume()

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open MacBook camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press q or Esc to quit.")

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            start_time = time.time()

            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            # Mirror webcam view.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(time.time() * 1000)

            result = recognizer.recognize_for_video(
                mp_image,
                timestamp_ms,
            )

            detected_hands = set()

            if result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    hand_name = get_handedness(result, hand_idx)

                    if hand_name not in ("Left", "Right"):
                        continue

                    detected_hands.add(hand_name)

                    static_label, score = get_static_gesture_for_hand(
                        result,
                        hand_idx,
                    )

                    hand_center_x = get_hand_center_x(
                        landmarks
                    )

                    # -----------------------------------
                    # Wave detection ONLY for open palm
                    # -----------------------------------

                    if static_label == "Open Palm":
                        wave_x_buffers[hand_name].append(
                            hand_center_x
                        )

                        if is_wave_motion(
                            wave_x_buffers[hand_name]
                        ):
                            prediction_buffers[hand_name].append(
                                "Wave"
                            )
                        else:
                            prediction_buffers[hand_name].append(
                                static_label
                            )

                    else:
                        wave_x_buffers[hand_name].clear()

                        prediction_buffers[hand_name].append(
                            static_label
                        )

                    current_labels[hand_name] = smooth_prediction(
                        prediction_buffers[hand_name]
                    )

                    current_scores[hand_name] = score

            # -----------------------------------
            # Handle missing hands
            # -----------------------------------

            for hand_name in ("Left", "Right"):
                if hand_name not in detected_hands:
                    wave_x_buffers[hand_name].clear()

                    prediction_buffers[hand_name].append(
                        NO_GESTURE
                    )

                    current_labels[hand_name] = smooth_prediction(
                        prediction_buffers[hand_name]
                    )

                    current_scores[hand_name] = 0.0

            # -----------------------------------
            # Gesture-Based macOS Controls
            # -----------------------------------

            current_time = time.time()

            if current_time - last_control_time > CONTROL_COOLDOWN:

                # LEFT HAND -> MUSIC CONTROL

                if ((current_labels["Left"] == "Peace Sign") or (current_labels["Left"] == "Pointer")):
                    pause_track()
                    last_control_time = current_time

                elif current_labels["Left"] == "Open Palm":
                    play_track()
                    last_control_time = current_time

                elif (
                    current_labels["Left"] == "Rockstar"
                    and current_time - last_track_skip_time > TRACK_SKIP_COOLDOWN
                ):
                    prev_track()
                    last_track_skip_time = current_time
                    last_control_time = current_time

            # Volume can repeat rapidly while held for quick multi-notch changes.
            if current_time - last_control_time > VOLUME_CONTROL_COOLDOWN:
                # RIGHT HAND -> VOLUME
                if current_labels["Right"] == "Open Palm":
                    current_volume = get_current_volume()
                    current_volume += VOLUME_STEP

                    current_volume = set_volume(
                        current_volume
                    )

                    last_control_time = current_time

                elif current_labels["Right"] == "Fist":
                    current_volume = get_current_volume()
                    current_volume -= VOLUME_STEP

                    current_volume = set_volume(
                        current_volume
                    )

                    last_control_time = current_time

            if (
                current_labels["Right"] == "Rockstar"
                and current_time - last_track_skip_time > TRACK_SKIP_COOLDOWN
                and current_time - last_control_time > CONTROL_COOLDOWN
            ):
                next_track()
                last_track_skip_time = current_time
                last_control_time = current_time

            # -----------------------------------
            # Drawing
            # -----------------------------------

            draw_landmarks(frame, result)

            # Optional:
            # draw_inter_hand_connections(frame, result)

            draw_hand_labels_near_hands(
                frame,
                result,
                current_labels,
                current_scores,
            )

            elapsed = time.time() - start_time

            if elapsed > 0:
                fps_buffer.append(1.0 / elapsed)

            avg_fps = (
                sum(fps_buffer) / len(fps_buffer)
                if fps_buffer else 0.0
            )

            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Music: Play/Pause Enabled",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Volume: {current_volume}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Live Gesture Classifier",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                pause_track()
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Camera released.")

if __name__ == "__main__":
    main()