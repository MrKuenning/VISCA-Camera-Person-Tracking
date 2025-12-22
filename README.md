# VISCA Camera Video Tracking Application

A professional PyQt6 application for controlling VISCA-compatible PTZ cameras with intelligent face tracking.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Professional UI** - Modern PyQt6 interface with responsive controls
- **VISCA Camera Control** - Full pan/tilt/zoom control via VISCA over IP
- **RTSP Video Streaming** - Live video feed from your PTZ camera
- **Intelligent Face Tracking** - Automatic face detection and smooth camera tracking
- **Multiple Detection Modes** - OpenCV Haar Cascade and DLib HOG+SVM detectors
- **Body Detection Fallback** - Track speakers even when face isn't visible
- **Tracking Persistence** - Maintains tracking when face is temporarily obscured
- **Manual Controls** - Full manual camera control with directional buttons

## Quick Start

### Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux
- VISCA-compatible PTZ camera (e.g., Sony SRG-300SE)
- Network connection to camera

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your camera:**
   - Edit `config/cameras.json` with your camera details
   - Or use the "Manage" button in the app to add cameras

### Launching the Application

**Option 1: Double-Click (Windows)**
```
Double-click: launch_app.bat
```

**Option 2: Python Command**
```bash
python launch_app.py
```

**Option 3: Direct Launch**
```bash
cd src
python main_app.py
```

## Usage

1. **Connect Camera:**
   - Select your camera from the dropdown
   - Click "Connect Camera"
   - Wait for green status indicator

2. **Start Video Stream:**
   - Verify RTSP URL is correct
   - Click "Connect Stream"
   - Video should appear in the main window

3. **Enable Tracking:**
   - Check "Preview Tracking" to see detection boxes
   - Check "Start Tracking Camera Control" to enable automatic camera movement
   - Adjust tracking options as needed

## Configuration

Camera settings are stored in `config/cameras.json`:

```json
{
  "cameras": [
    {
      "name": "Main Camera",
      "ip": "10.1.2.8",
      "control_port": 52381,
      "stream_url": "rtsp://10.1.2.8/video2",
      "inverted": false
    }
  ]
}
```

## Project Structure

```
App1/
├── launch_app.py          # Main launcher script
├── launch_app.bat         # Windows batch launcher
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── src/                   # Source code
│   ├── main_app.py        # Main application (PyQt6)
│   ├── face_tracker.py    # Face/body detection and tracking logic
│   └── camera_manager.py  # Camera configuration management
├── config/                # Configuration files
│   └── cameras.json       # Camera settings
├── docs/                  # Documentation
│   ├── Design Goals.txt   # Original design specifications
│   └── TRACKING_FEATURES.md
└── legacy/                # Old Tkinter version (deprecated)
    └── Video_Tracking_App.py
```

## Technical Architecture

### Libraries & Dependencies
The application relies on several robust libraries to deliver professional performance:

- **[PyQt6](https://pypi.org/project/PyQt6/)**: Core GUI framework for the modern application window and controls.
- **[OpenCV (opencv-python)](https://pypi.org/project/opencv-python/)**: Handles RTSP video stream decoding and image manipulation.
- **[MediaPipe](https://pypi.org/project/mediapipe/)**: Provides high-performance, real-time face and pose detection for the "Smooth Tracker".
- **[VISCA-over-IP](https://pypi.org/project/visca-over-ip/)**: Manages network communication with PTZ cameras using the standard VISCA protocol.
- **[Simple-PID](https://pypi.org/project/simple-pid/)**: Implements PID (Proportional-Integral-Derivative) controllers for smooth, natural camera movements.
- **[NumPy](https://pypi.org/project/numpy/)**: Used for efficient array operations and coordinate calculations.

### Key Modules & Functions

#### `src/main_app.py`
The entry point and main GUI controller.
- **`VideoTrackingApp`**: The main widget class inheriting from `QMainWindow`. It manages the 3-column UI layout, event handling, and coordination between the camera manager, video stream, and tracker.
- **`VideoThread`**: A `QThread` subclass that handles RTSP video capture in a background thread to ensure the UI remains responsive.
- **`process_frame(frame)`**: The central processing loop called for every new video frame. It triggers detection, updates the tracker control loop, and draws visualization overlays.

#### `src/smooth_tracker.py`
The modern tracking engine using detection smoothing and PID control.
- **`SmoothTracker`**: The primary tracking class. It encapsulates the detection logic and the movement control loop.
- **`PositionSmoother`**: Implements an Exponential Moving Average (EMA) filter to significantly reduce jitter in detection coordinates.
- **`SimplePIDController`**: Calculates the precise camera speed needed to center the subject based on the current error (distance from center), utilizing P, I, and D terms for smooth acceleration and deceleration.
- **`detect_faces(frame)`**: Wrapper around MediaPipe's Face Detection to reliably find subjects in the video frame.

#### `src/camera_manager.py`
Handles persistence and management of camera settings.
- **`CameraManager`**: Loads and saves camera connection details (IP, Port, URL) to `config/cameras.json`.
- **`CameraEditorDialog`**: A custom dialog UI for adding, editing, and removing camera configurations.

#### `src/face_tracker.py` (Legacy)
Contains older detection algorithms, primarily preserved for fallback or specific use cases.
- **`FaceTracker`**: Original tracking implementation using OpenCV Haar Cascades and threshold-based movement logic.


## Tracking Features

### Detection Modes
- **OpenCV Face Tracking** - Fast, reliable Haar Cascade detector
- **DLib Face Tracking** - More accurate HOG+SVM detector (slower)

### Advanced Options
- **Use Body Tracking** - Fallback to body detection when face not visible
- **Require Face & Body** - Only move camera when both are detected
- **Tracking Persistence** - Continue tracking briefly when target is lost

### Performance Tuning
- **Detection Interval** - How often to run face detection (0.1-1.0s)
- **Detection Resolution** - Scale factor for detection (0.2-1.0)
- **Camera Speed** - Movement speed for tracking (0.1-10.0)

## Troubleshooting

**Camera won't connect:**
- Verify camera IP address and port
- Check network connectivity
- Ensure camera is powered on
- Try pinging the camera IP

**Video stream not working:**
- Verify RTSP URL is correct
- Check camera stream settings
- Ensure firewall isn't blocking RTSP (port 554)

**Tracking too jerky:**
- Increase detection interval
- Lower detection resolution
- Adjust camera speed

**Missing dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

## Legacy Version

The original Tkinter version is preserved in `legacy/Video_Tracking_App.py` but is **deprecated**. Use the PyQt6 version for better performance and UI.

## Support

For issues or questions, refer to the documentation in the `docs/` folder.

## License

MIT License - See LICENSE file for details
