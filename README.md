# Driver Safety Monitoring System

## Abstract

This project implements a real-time driver safety monitoring web application that uses a YOLOv11 deep learning model to analyze live webcam or recorded video streams and classify driver behaviors into six categories: awake, distracted, eyes_closed, phone, smoking, and yawn. The system continuously tracks these behaviors, applies frame-based thresholds to detect risky patterns such as prolonged eye closure, repeated yawning, distraction, phone usage, and smoking, and then triggers contextual audio/visual alerts to warn the driver. For critical violations (eyes_closed, phone, and smoking), the application captures evidence images and automatically sends email notifications containing timestamps and violation details, supporting both individual safety and fleet-level audit logging. The frontend provides a modern, production-ready dashboard for starting webcam detection or uploading videos, viewing live detections and alert status, and configuring email settings.

## Features

- **Real-time Detection**: Continuous monitoring using YOLOv11 model
- **Behavioral Alerts**: Context-aware alerts for yawn, eyes closed, distraction, phone, and smoking
- **Email Notifications**: Automatic violation logging with evidence images
- **Modern UI**: Professional React-based frontend with Tailwind CSS
- **Webcam & Video Support**: Works with live webcam or uploaded video files

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model file `best (2).pt` is in the project root directory.

## Usage

1. Start the Flask backend server:
```bash
cd backend
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Configure email settings (optional) by clicking "Email Config" button.

4. Start detection by either:
   - Clicking "Start Webcam Detection" for live camera feed
   - Clicking "Upload Video" to process a video file

## Alert System

- **Yawn (3+ consecutive)**: Visual/audio alarm with manual stop button (auto-stops after 10s)
- **Eyes Closed (3+ seconds)**: Alarm + email with evidence image
- **Distracted (4+ seconds)**: Voice alert ("Please do not get distracted") until awake
- **Phone Detected**: Voice alert ("Please do not use phone") + email with evidence
- **Smoking Detected**: Voice alert ("Please do not smoke inside car") + email with evidence

## Email Configuration

Configure SMTP settings in the UI:
- SMTP Server (default: smtp.gmail.com)
- SMTP Port (default: 587)
- Sender Email & Password
- Recipient Email

For Gmail, you may need to use an "App Password" instead of your regular password.

## Model Classes

The model detects 6 classes:
- awake
- distracted
- eyes_closed
- phone
- smoking
- yawn

## Project Structure

```
.
├── backend/
│   ├── app.py              # Flask backend server
│   └── templates/
│       └── index.html      # Frontend React app
├── best (2).pt            # YOLOv11 model file
├── data (1).yaml          # Model configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Notes

- Detection thresholds are frame-based (not time-based)
- Voice alerts use browser's Web Speech API
- Violation images are saved in `violations/` directory
- Uploaded videos are saved in `uploads/` directory

