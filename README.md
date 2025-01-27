# Gesture Volume Control

Control your system volume with hand gestures using computer vision technology. This application allows you to adjust your computer's volume by measuring the distance between your thumb and index finger.

![Gesture Control Demo](https://your-demo-gif-here.gif)

## ✨ Features

- Real-time hand gesture recognition
- Visual volume level indicator
- Intuitive gesture control interface
- Live camera feed with hand tracking visualization
- System volume integration (currently supports macOS)
- Smooth volume adjustment with natural hand movements

## 🔧 Requirements

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- macOS for volume control functionality (Windows support coming soon)

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/ChaseVanacora/HandVolume.git
cd HandVolume
```

2. Install required dependencies:
```bash
pip install -r requirements/requirements.txt
```

## 🚀 Usage

1. Run the application:
```bash
python main.py
```

2. Position your hand in front of your camera
3. Use your thumb and index finger to control the volume:
   - Move fingers apart to increase volume
   - Bring fingers closer to decrease volume
4. Press 'q' to quit the application

## 💡 How It Works

The application uses:
- **MediaPipe** for real-time hand landmark detection
- **OpenCV** for camera feed processing and visualization
- Custom algorithms for gesture-to-volume mapping
- System-specific volume control integration

## 🎯 Technical Details

- Hand tracking confidence threshold: 70%
- Volume range: 0-100%
- Camera feed is flipped horizontally for natural interaction
- Volume adjustment is mapped to finger distance range of 30-350 pixels
- Green visualization includes:
  - Hand landmark connections
  - Volume level bar
  - Finger distance line
  - Usage instructions

