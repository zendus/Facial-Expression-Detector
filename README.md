# Real-time Facial Expression Detector with Emoji Replacement

## Overview

This project utilizes deep learning techniques to build a real-time facial expression detector using Keras and OpenCV. The system can identify facial expressions such as happiness, sadness, anger, etc., and overlays corresponding emoji equivalents on the detected faces in real-time.

## Features

- **Facial Expression Detection:** Utilizes a pre-trained deep learning model to recognize facial expressions.
- **Real-time Processing:** Processes video frames from the webcam in real-time, providing instant feedback.
- **Emoji Replacement:** Replaces detected faces and expressions with emoji equivalents, enhancing the visual output.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/zendus/Facial-Expression-Detector.git
   cd Facial-Expression-Detector
   ```

2. **Install Dependencies:**
   ```bash
   pip install opencv-python tensorflow numpy
   ```

3. **Download Pre-trained Model:**
   Download the pre-trained model from [fer2013_mini_XCEPTION.119-0.65.hdf5](https://github.com/priya-dwivedi/face_and_emotion_detection/blob/master/emotion_detector_models/fer2013_mini_XCEPTION.119-0.65.hdf5) and place it in the project directory.

4. **Run the Script:**
   ```bash
   python app.py
   ```

## Dependencies

- OpenCV
- Tensorflow
- NumPy

## Notes

- Ensure your webcam is properly connected and accessible.
- Adjust paths and parameters as needed for your setup.
- Real-time video processing may be resource-intensive; ensure your system can handle it effectively.

## Contributing

Contributions are welcome! If you have suggestions, enhancements, or bug fixes, feel free to open an issue or create a pull request.

