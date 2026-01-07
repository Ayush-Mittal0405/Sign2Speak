# 🖐️ Sign Language Detection with Multi-Language Voice Output

A real-time **Sign Language Detection System** built using **YOLO (Ultralytics)** that recognizes hand gestures through a webcam and converts them into **spoken audio output** in multiple regional languages.

This project is designed to improve **accessibility and communication** for speech-impaired and hearing-impaired individuals.

---

## 🚀 Features

- 🎯 Real-time sign language detection using webcam
- 🔊 Voice output for detected signs
- 🌐 Multi-language support  
  - English  
  - Hindi  
  - Gujarati
- 🔁 Instant language switching  
  - `SPACE` key (PC / Laptop)  
  - Physical button (Raspberry Pi GPIO)
- ⏱️ Audio cooldown to avoid repeated speech
- 💻 Works on **PC and Raspberry Pi**

---

## 🧠 How It Works

1. Webcam captures live video
2. YOLO model detects hand gestures
3. Detected class label is identified
4. Corresponding audio file is played
5. User can switch output language instantly

---

## 🛠️ Tech Stack Used

### Programming Language
- Python 🐍

### Machine Learning & Computer Vision
- YOLO (Ultralytics)
- OpenCV

### Audio Handling
- Pygame
  
---

## 📦 Required Libraries

```txt
ultralytics
pygame
opencv-python
numpy

