# ShotTracker

## Team Information

* **Project Name:** ShotTracker

## Members

* Adam Pinkos (email: [ap971723@ohio.edu], [pinkos2003@gmail.com], gh: [pinkos1](https://github.com/pinkos1))

## About this project

ShotTracker is a Python-based desktop application that allows users to upload basketball videos and automatically detect the basketball, hoop, and players using YOLOv8. The system plays the video back at an accelerated speed and overlays detections, with the goal of tracking shot makes and misses.

## Platform

* Platforms: Windows, macOS, WSL
* Language: Python 3.12+

## Frameworks/Tools

* IDE: Visual Studio Code
* Version Control: Git
* GUI: Tkinter
* Libraries: OpenCV, Ultralytics (YOLOv8), Doxygen

## How to build/run

### 1. Install dependencies

Install required packages using pip:

```bash
pip install opencv-python ultralytics
```

Tkinter is typically bundled with Python on Windows/macOS. On Linux:

```bash
sudo apt-get install python3-tk
```

### 2. Run the application

Ensure your file structure looks like:

```
ShotTracker/
├── gui.py
├── video_reader.py
├── README.md
```

Then run the GUI:

```bash
python gui.py
```

## Usage

* Click "Upload Video" in the GUI
* Select a video file (e.g., `.mp4`, `.avi`, `.mov`)
* The video will play at 10x speed with bounding boxes drawn on basketball detections

## File Descriptions

| File              | Description                              |
| ----------------- | ---------------------------------------- |
| `gui.py`          | Tkinter-based interface for video upload |
| `video_reader.py` | Reads video and applies YOLOv8 detection |
| `README.md`       | This file                                |

## Future Improvements

* Add hoop and player detection
* Determine shot makes and misses algorithmically
* Export shot data and analytics
* Enhance GUI with video controls and stats display

## Documentation

To generate documentation from Doxygen comments:

```bash
doxygen Doxyfile
```

If using a Makefile:

```bash
make docs
```

