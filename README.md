# AI Yoga Pose Trainer (Real-Time Pose Correction System)

## Description
This project is an AI-powered real-time yoga pose correction system designed for guiding users through Surya Namaskar (12 poses). It uses TensorFlow MoveNet to detect human body keypoints from live webcam input and analyzes posture using joint angle calculations.

The system compares user pose angles with precomputed reference angles and provides instant corrective feedback such as “bend knee more” or “raise arm”.

## Key Features
- Real-time human pose detection using TensorFlow MoveNet  
- Supports complete Surya Namaskar sequence (12 poses)  
- Joint angle-based posture analysis  
- Dynamic feedback system for pose correction  
- Real-time accuracy and stability scoring  
- Pose hold detection with repetition tracking  
- Smooth tracking using rolling average and stability metrics  
- Intelligent framing guidance for proper camera positioning  

## Tech Stack
- JavaScript  
- TensorFlow.js  
- MoveNet (Pose Detection Model)  
- HTML, CSS  
- Canvas API  

## How It Works
1. The system captures live video using webcam  
2. MoveNet detects body keypoints in real time  
3. Joint angles (elbow, knee, shoulder, hip) are calculated  
4. Angles are compared with reference pose data  
5. Accuracy and stability scores are computed  
6. Feedback is generated to correct posture  
7. Pose is locked when held correctly for a fixed duration  

## Reference Data
Reference angles for each pose were generated using a custom extraction script that processes pose images and computes mean joint angles.

## Dataset
The model uses TensorFlow MoveNet (pre-trained), and additional reference data was generated from a custom dataset of yoga pose images.  
Due to size limitations, the dataset is not included in this repository.

## How to Run
1. Clone the repository  
2. Open the project folder  
3. Run `index.html` in a browser  
4. Allow camera access  
5. Start the guided yoga session  

## Future Improvements
- Add more yoga sequences and poses  
- Improve accuracy using custom-trained models  
- Add performance analytics and progress tracking  
- Mobile responsiveness and app deployment  
