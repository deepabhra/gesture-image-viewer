# 🖼️ Gesture-Controlled Image Viewer

Welcome to the future of no-touch image browsing 👋✨  
This Python project lets you **view, zoom, delete, and switch images** with just your **hand gestures**, using your webcam and some OpenCV magic. No clicks. No keys. Just vibes.

## 🔮 Features

- 🖼️ **Image Viewing** from a folder
- 🤏 **Zoom In/Out** with pinch gestures on both hands
- ⬅️➡️ **Next/Previous Image** navigation using hand flicks
- 🗑️ **Delete Image** with a fist (right hand)
- ✋ **Exit App** with a closed fist (left hand)
- 💬 Real-time gesture feedback printed in both terminal & webcam window
- 🧠 Gesture cooldown + smoothing buffer to avoid accidental triggers

## 🎯 How It Works

- Uses `cvzone.HandTrackingModule` + OpenCV to track your hand landmarks.
- Detects specific gestures based on finger positions.
- Displays zoomed-in images on a centered canvas.
- Webcam feed shows gesture info and debug messages.

## 🚀 Setup Instructions

### 1. Clone This Repo

```bash
git clone https://github.com/deepabhra/gesture-image-viewer.git
cd gesture-image-viewer
```
## Install Requirements
- Make sure you're using Python 3.7+
```bash
pip install opencv-python numpy cvzone
```
## Add Your Images
- Drop your `.jpg`, `.png`, `.jpeg`, `.bmp`, `.gif` images in the `resources` folder.

## Run It
```bash
python image_viewer.py
```

## ✋ Gesture Controls
| Gesture                   | Hand       | Action         |
| ------------------------- | ---------- | -------------- |
| ✊ Fist (all fingers down) | Right Hand | Delete image    |
| ✊ Fist (all fingers down) | Left Hand  | Exit viewer   |
| 🤏 Pinch fingers inward   | Both Hands | Zoom out       |
| 🤏 Spread fingers out     | Both Hands | Zoom in        |
| 👋 4 fingers up           | Left Hand  | Previous image |
| 👋 4 fingers up           | Right Hand | Next image     |

## 💡 To-Do & Future Upgrades
- ✅ Support for image viewing ✅
- Add support for video files (gesture-controlled playback)

## Built with 
- [OpenCV](https://opencv.org/)
- [cvzone](https://github.com/cvzone)

## 👨‍💻 Author
**Abhradeep**, a 3rd-year Computer Science student at Visva Bharati University.
Building the future, one gesture at a time.

Connect with me:\
[📸 Instagram](https://www.instagram.com/deep_abhra/)

## 📄 Project Report

This project was developed as my final-year B.Sc. Computer Science project at Visva-Bharati University.

The complete project report includes:
- Project overview
- System architecture
- Methodology
- Workflow
- Code explanation
- Results & Output
- Future Scope
- Conclusion
- References

📘 **Read the full report here:**
[Project Report (PDF)](./Air%20Gesture%20Controlled%20Media%20Viewer%20Using%20Webcam%20and%20OpenCV.pdf)

## ⚠️ Disclaimer / Warning

This code is designed to work with standard webcams, which typically flip the image horizontally (mirror view). If your camera does not flip the image, the hand detection logic may be reversed: your right hand will be detected as the left hand, and your left hand as the right hand. In such cases, please assume that the gestures for the right and left hands are swapped.