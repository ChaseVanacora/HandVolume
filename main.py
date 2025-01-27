import cv2
import mediapipe as mp
import math
import numpy as np
import sys
import platform
import subprocess

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

# Initialize volume range
min_vol = 0
max_vol = 100


def set_macos_volume(volume_percentage):
    """Set system volume on macOS using osascript"""
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {volume_percentage}"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error setting volume: {e}")


# Initialize video capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open camera")
except Exception as e:
    print(f"Error: Could not access the camera - {e}")
    sys.exit(1)

# Set initial window position
cv2.namedWindow("Gesture Volume Control")
cv2.moveWindow("Gesture Volume Control", 40, 30)


def draw_volume_bar(img, vol_percentage, x=50, y_top=150, y_bottom=400, width=35):
    """Draw the volume visualization bar"""
    # Draw outline
    cv2.rectangle(img, (x, y_top), (x + width, y_bottom), (0, 255, 0), 3)
    # Draw filled portion
    vol_height = int(np.interp(vol_percentage, [0, 100], [y_bottom, y_top]))
    cv2.rectangle(img, (x, vol_height), (x + width, y_bottom), (0, 255, 0), cv2.FILLED)
    # Draw volume percentage
    cv2.putText(
        img,
        f"{int(vol_percentage)}%",
        (x, y_top - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip image horizontally for selfie-view
        img = cv2.flip(img, 1)

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get thumb and index finger landmarks
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

                # Calculate distance between fingers
                x1, y1 = int(thumb.x * img.shape[1]), int(thumb.y * img.shape[0])
                x2, y2 = int(index.x * img.shape[1]), int(index.y * img.shape[0])
                length = math.hypot(x2 - x1, y2 - y1)

                # Draw line between thumb and index finger
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Map finger distance to volume percentage (0-100)
                # Using wider range for less sensitivity and requiring more finger separation for max volume
                vol_percentage = np.interp(length, [30, 350], [0, 100])
                vol_percentage = np.clip(
                    vol_percentage, 0, 100
                )  # Ensure volume stays within bounds

                # Set system volume
                if platform.system() == "Darwin":  # macOS
                    set_macos_volume(vol_percentage)
                elif platform.system() == "Windows":
                    print("Windows volume control not implemented")

                # Draw volume visualization
                draw_volume_bar(img, vol_percentage)

        # Add instructions text
        cv2.putText(
            img,
            "Move thumb and index finger to control volume",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img,
            "Press 'q' to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Show the image
        cv2.imshow("Gesture Volume Control", img)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
