# Camera Vision Controller

`comp_vision_controller.py` is a macOS gesture controller that uses MediaPipe and OpenCV to recognize hand gestures from your webcam and map them to media controls.

## What it does

- Detects up to two hands in the webcam feed
- Recognizes static gestures such as:
  - `Open Palm`
  - `Fist`
  - `Pointing Up`
  - `Victory` (Peace Sign)
  - `ILoveYou` (Rockstar)
  - `Thumbs Up` / `Thumbs Down`
- Detects an additional `Wave` motion when an open palm moves left/right
- Uses gestures to control macOS audio playback and volume for a configured music app

## Requirements

- Python 3.10+ (or compatible)
- `opencv-python`
- `mediapipe`
- macOS with `osascript` available
- A valid MediaPipe gesture model file at `gesture_recognizer.task`

## Setup

1. Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install opencv-python mediapipe
```

3. Ensure the gesture model file is present:

```bash
ls gesture_recognizer.task
```

## Usage

Run the controller from the project folder:

```bash
python comp_vision_controller.py
```

Press `q` or `Esc` to quit.

## Gesture controls

### Left hand
- `Open Palm` → Play music
- `Peace Sign` or `Pointer` → Pause music
- `Rockstar` (`ILoveYou`) → Previous track

### Right hand
- `Open Palm` → Increase volume
- `Fist` → Decrease volume
- `Rockstar` (`ILoveYou`) → Next track

## Configuration

- Change the target music app by updating `MUSIC_APP` at the top of the script.
- Toggle mirrored handedness behavior by updating `SWAP_HANDEDNESS`.
- Adjust volume or gesture cooldowns by changing constants near the top of the script.

## Notes

- This script is designed for macOS and relies on AppleScript via `osascript`.
- The webcam feed is mirrored so left/right gestures correspond naturally to the user.
- The script draws hand landmarks and labels directly on the OpenCV window.
