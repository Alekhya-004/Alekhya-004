from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import threading
import time
from collections import deque
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import os
import platform
import json
from urllib import parse, request as urllib_request

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_FILE = 'best (2).pt'
DETECTION_CONFIDENCE = 0.25
WEBCAM_INDEX = 0

DEFAULT_EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'chinthalaalekhya1@gmail.com',
    'sender_password': 'wnwo fovq mutg ikjs',
    'recipient_email': 'valekhya55@jnn.edu.in'
}

# Eyes closed: 4 seconds (time-based). Frames at ~30fps = 120.
EYES_CLOSED_SECONDS = 4
THRESHOLDS = {
    'yawn_consecutive': 3,
    'eyes_closed_seconds': EYES_CLOSED_SECONDS,
    'distracted_frames': 120,
    'yawn_alarm_duration': 10
}
BASE_THRESHOLDS = THRESHOLDS.copy()

DEFAULT_MOBILE_CONFIG = {
    'enabled': False,
    'webhook_url': ''
}

DEFAULT_WEATHER_CONFIG = {
    'enabled': False,
    'api_key': '',
    'city': '',
    'country_code': 'IN',
    'rain_sensitivity_multiplier': 0.8,   # lower threshold => more sensitive
    'night_sensitivity_multiplier': 0.75  # lower threshold => more sensitive
}

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Alarm sound: full path to WAV file (used when serving /sounds/mixkit-space-shooter-alarm-1002.wav)
ALARM_WAV_PATH = r'C:\Users\DELL\Desktop\Driver Drowsiness\mixkit-space-shooter-alarm-1002.wav'

# ============================================================================
# END CONFIGURATION
# ============================================================================

model_path = os.path.join(PROJECT_ROOT, MODEL_FILE)
model = YOLO(model_path)  # loaded once at startup; used for prediction in _detection_worker
class_names = ['awake', 'distracted', 'eyes_closed', 'phone', 'smoking', 'yawn']

camera = None
video_source = None
is_detecting = False
current_frame = None
annotated_frame = None  # latest YOLO-annotated frame from detection worker
detection_lock = threading.Lock()
frame_lock = threading.Lock()  # protects current_frame / annotated_frame between threads
detection_thread = None
fps_estimate = 30.0  # will be updated from camera if available

state_tracker = {
    'yawn_count': 0,
    'eyes_closed_frames': 0,
    'eyes_closed_start_time': None,
    'distracted_frames': 0,
    'phone_detected': False,
    'smoking_detected': False,
    'last_detection': None,
    'last_confidence': 0.0
}

alert_states = {
    'yawn_alarm': False,
    'yawn_alarm_start': None,
    'eyes_closed_alarm': False,
    'eyes_closed_alert': False,
    'eyes_closed_email_sent': False,
    'distracted_alert': False,
    'phone_alert': False,
    'phone_email_sent': False,
    'smoking_alert': False,
    'smoking_email_sent': False
}

email_config = DEFAULT_EMAIL_CONFIG.copy()
mobile_config = DEFAULT_MOBILE_CONFIG.copy()
weather_config = DEFAULT_WEATHER_CONFIG.copy()
frame_history = deque(maxlen=300)
notification_feed = deque(maxlen=100)
env_context = {
    'is_night': False,
    'is_raining': False,
    'weather_description': 'Unknown',
    'last_weather_update': None
}
last_weather_fetch_ts = 0.0


def open_front_facing_camera(max_index=3):
    """
    Try to find a usable webcam by probing indices 0..max_index.
    Prefers a camera that actually returns non‑black frames.
    """
    for idx in range(max_index + 1):
        if platform.system() == "Windows":
            cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(idx)

        if not cam.isOpened():
            cam.release()
            continue

        # Give the camera a brief moment to warm up
        time.sleep(0.3)
        ok, frame = cam.read()
        if not ok or frame is None:
            cam.release()
            continue

        # Skip cameras that only return (almost) pure black frames
        if frame.mean() < 2:
            cam.release()
            continue

        return cam, idx, frame

    return None, None, None


def detect_objects(frame):
    results = model.predict(frame, conf=DETECTION_CONFIDENCE, verbose=False)
    return results[0]


def is_night_time():
    hour = datetime.now().hour
    return hour >= 19 or hour < 6


def update_weather_context():
    """
    Refresh rain context from weather API every 10 minutes.
    If config is disabled/not set, context defaults to non-rain.
    """
    global last_weather_fetch_ts
    now = time.time()
    if now - last_weather_fetch_ts < 600:
        return

    last_weather_fetch_ts = now
    env_context['is_night'] = is_night_time()

    if not weather_config.get('enabled'):
        env_context['is_raining'] = False
        env_context['weather_description'] = 'Weather API disabled'
        env_context['last_weather_update'] = datetime.now().isoformat()
        return

    api_key = weather_config.get('api_key', '').strip()
    city = weather_config.get('city', '').strip()
    if not api_key or not city:
        env_context['is_raining'] = False
        env_context['weather_description'] = 'Weather API not configured'
        env_context['last_weather_update'] = datetime.now().isoformat()
        return

    country_code = weather_config.get('country_code', 'IN').strip()
    query = parse.urlencode({'q': f'{city},{country_code}', 'appid': api_key, 'units': 'metric'})
    weather_url = f'https://api.openweathermap.org/data/2.5/weather?{query}'
    try:
        with urllib_request.urlopen(weather_url, timeout=8) as response:
            payload = json.loads(response.read().decode('utf-8'))

        weather_items = payload.get('weather', [])
        weather_main = weather_items[0].get('main', '') if weather_items else ''
        weather_description = weather_items[0].get('description', 'Unknown') if weather_items else 'Unknown'
        rain_volume = float(payload.get('rain', {}).get('1h', 0) or 0)

        env_context['is_raining'] = weather_main.lower() in ('rain', 'drizzle', 'thunderstorm') or rain_volume > 0
        env_context['weather_description'] = weather_description
        env_context['last_weather_update'] = datetime.now().isoformat()
    except Exception as e:
        env_context['is_raining'] = False
        env_context['weather_description'] = f'Weather API error: {str(e)}'
        env_context['last_weather_update'] = datetime.now().isoformat()


def get_active_thresholds():
    """
    Dynamic sensitivity:
    - At night and in rain, lower thresholds to alert earlier.
    """
    update_weather_context()
    active = BASE_THRESHOLDS.copy()
    multiplier = 1.0
    if env_context['is_night']:
        multiplier *= weather_config.get('night_sensitivity_multiplier', 0.75)
    if env_context['is_raining']:
        multiplier *= weather_config.get('rain_sensitivity_multiplier', 0.8)

    # Keep safe lower bounds
    active['yawn_consecutive'] = max(1, int(round(BASE_THRESHOLDS['yawn_consecutive'] * multiplier)))
    active['eyes_closed_seconds'] = max(2, round(BASE_THRESHOLDS['eyes_closed_seconds'] * multiplier, 1))
    active['distracted_frames'] = max(30, int(round(BASE_THRESHOLDS['distracted_frames'] * multiplier)))
    return active


def update_state_tracker(detections):
    detected_classes = []
    if detections.boxes is not None and len(detections.boxes) > 0:
        for i in range(len(detections.boxes)):
            cls = int(detections.boxes.cls[i])
            conf = float(detections.boxes.conf[i])
            class_name = class_names[cls]
            detected_classes.append((class_name, conf))

    with detection_lock:
        if detected_classes:
            main_detection = max(detected_classes, key=lambda x: x[1])
            state_tracker['last_detection'] = main_detection[0]
            state_tracker['last_confidence'] = main_detection[1]
        else:
            state_tracker['last_detection'] = 'awake'
            state_tracker['last_confidence'] = 0.0

        frame_history.append(state_tracker['last_detection'])

        if state_tracker['last_detection'] == 'yawn':
            state_tracker['yawn_count'] += 1
        else:
            state_tracker['yawn_count'] = 0

        if state_tracker['last_detection'] == 'eyes_closed':
            state_tracker['eyes_closed_frames'] += 1
            if state_tracker['eyes_closed_start_time'] is None:
                state_tracker['eyes_closed_start_time'] = time.time()
        else:
            state_tracker['eyes_closed_frames'] = 0
            state_tracker['eyes_closed_start_time'] = None

        if state_tracker['last_detection'] == 'distracted':
            state_tracker['distracted_frames'] += 1
        else:
            state_tracker['distracted_frames'] = 0

        state_tracker['phone_detected'] = any(c[0] == 'phone' for c in detected_classes)
        state_tracker['smoking_detected'] = any(c[0] == 'smoking' for c in detected_classes)


def eyes_closed_duration_seconds():
    """Return how many seconds eyes have been closed. Safe to call WITHOUT detection_lock held."""
    with detection_lock:
        if state_tracker['eyes_closed_start_time'] is None:
            return 0.0
        return time.time() - state_tracker['eyes_closed_start_time']


def check_alerts():
    global alert_states
    active_thresholds = get_active_thresholds()
    with detection_lock:
        if state_tracker['yawn_count'] >= active_thresholds['yawn_consecutive']:
            if not alert_states['yawn_alarm']:
                alert_states['yawn_alarm'] = True
                alert_states['yawn_alarm_start'] = time.time()
                push_notification('yawn', 'warning', 'Frequent yawning detected')
        else:
            if alert_states['yawn_alarm']:
                alert_states['yawn_alarm'] = False
                alert_states['yawn_alarm_start'] = None

        if alert_states['yawn_alarm'] and alert_states['yawn_alarm_start']:
            if time.time() - alert_states['yawn_alarm_start'] >= active_thresholds['yawn_alarm_duration']:
                alert_states['yawn_alarm'] = False
                alert_states['yawn_alarm_start'] = None

        # Calculate inline — must NOT call eyes_closed_duration_seconds() here because
        # that function also acquires detection_lock and would deadlock the same thread.
        t = state_tracker['eyes_closed_start_time']
        closed_sec = (time.time() - t) if t is not None else 0.0

        if closed_sec >= active_thresholds['eyes_closed_seconds']:
            if not alert_states['eyes_closed_alert']:
                alert_states['eyes_closed_alert'] = True
                alert_states['eyes_closed_alarm'] = True
                push_notification('eyes_closed', 'critical', 'Eyes closed threshold breached')
            if not alert_states['eyes_closed_email_sent']:
                capture_and_email_violation('eyes_closed')
                alert_states['eyes_closed_email_sent'] = True
        else:
            alert_states['eyes_closed_alert'] = False
            alert_states['eyes_closed_email_sent'] = False
            alert_states['eyes_closed_alarm'] = False

        if state_tracker['distracted_frames'] >= active_thresholds['distracted_frames']:
            if not alert_states['distracted_alert']:
                push_notification('distracted', 'warning', 'Driver distraction sustained')
            alert_states['distracted_alert'] = True
        else:
            alert_states['distracted_alert'] = False

        if state_tracker['phone_detected']:
            if not alert_states['phone_alert']:
                push_notification('phone', 'critical', 'Phone usage detected while driving')
            alert_states['phone_alert'] = True
            if not alert_states['phone_email_sent']:
                capture_and_email_violation('phone')
                alert_states['phone_email_sent'] = True
        else:
            alert_states['phone_alert'] = False
            alert_states['phone_email_sent'] = False

        if state_tracker['smoking_detected']:
            if not alert_states['smoking_alert']:
                push_notification('smoking', 'warning', 'Smoking detected inside vehicle')
            alert_states['smoking_alert'] = True
            if not alert_states['smoking_email_sent']:
                capture_and_email_violation('smoking')
                alert_states['smoking_email_sent'] = True
        else:
            alert_states['smoking_alert'] = False
            alert_states['smoking_email_sent'] = False


def send_mobile_notification(notification):
    if not mobile_config.get('enabled'):
        return
    webhook_url = mobile_config.get('webhook_url', '').strip()
    if not webhook_url:
        return
    payload = json.dumps(notification).encode('utf-8')
    req = urllib_request.Request(
        webhook_url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib_request.urlopen(req, timeout=8):
            pass
    except Exception as e:
        print(f"[mobile-notification] failed: {e}")


def push_notification(event_type, severity, message):
    notification = {
        'event_type': event_type,
        'severity': severity,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'detection': state_tracker.get('last_detection', 'unknown'),
        'confidence': round(float(state_tracker.get('last_confidence', 0.0)), 4),
        'context': {
            'is_night': env_context['is_night'],
            'is_raining': env_context['is_raining'],
            'weather_description': env_context['weather_description']
        }
    }
    notification_feed.appendleft(notification)
    threading.Thread(target=send_mobile_notification, args=(notification,), daemon=True).start()


def capture_and_email_violation(violation_type):
    if current_frame is None:
        print("[Violation] Skip email: no frame")
        return
    if not email_config.get('sender_email') or not email_config.get('recipient_email'):
        print("[Violation] Skip email: sender or recipient not set")
        return
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_dir = os.path.join(PROJECT_ROOT, "violations")
        os.makedirs(violations_dir, exist_ok=True)
        img_path = os.path.join(violations_dir, f"{violation_type}_{timestamp}.jpg")
        cv2.imwrite(img_path, current_frame)
        threading.Thread(target=send_violation_email, args=(violation_type, img_path, timestamp), daemon=True).start()
    except Exception as e:
        print(f"Error capturing violation: {e}")


def send_violation_email(violation_type, img_path, timestamp):
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']
        msg['Subject'] = f"Driver Safety Alert: {violation_type.replace('_', ' ').title()}"

        body = f"""
Driver Safety Violation Detected

Violation Type: {violation_type.replace('_', ' ').title()}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Please review the attached evidence image.
"""
        msg.attach(MIMEText(body, 'plain'))
        with open(img_path, 'rb') as f:
            img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(img_path))
            msg.attach(image)

        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])
        server.send_message(msg)
        server.quit()
        print(f"[Violation] Email sent: {violation_type} -> {email_config['recipient_email']}")
    except Exception as e:
        print(f"Error sending email: {e}")


def _start_detection_thread():
    """Start the background thread that runs YOLO prediction on the live feed."""
    global detection_thread
    detection_thread = threading.Thread(target=_detection_worker, daemon=True)
    detection_thread.start()
    print("[app] Detection thread started — model is now predicting on live feed.")


def _detection_worker():
    """Background thread: run YOLO every 0.2s, store annotated frame, update state/alerts."""
    global current_frame, annotated_frame
    pred_count = 0
    while is_detecting:
        time.sleep(0.2)
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
        if frame is None:
            continue
        try:
            results = detect_objects(frame)
            drawn = results.plot()  # frame with bounding boxes drawn
            with frame_lock:
                annotated_frame = drawn
            update_state_tracker(results)
            check_alerts()
            pred_count += 1
            if pred_count % 25 == 0:
                with detection_lock:
                    print(f"[detection] predicting... {state_tracker['last_detection']} ({state_tracker['last_confidence']:.2f})")
        except Exception as e:
            import traceback
            print(f"[detection_worker] error: {e}")
            traceback.print_exc()


def generate_frames():
    """
    Stream webcam frames to the browser.
    Uses the YOLO-annotated frame (with bounding boxes) when available,
    falls back to the raw frame until the first detection is ready.
    """
    global current_frame, annotated_frame, camera, video_source
    frame_count = 0
    while is_detecting:
        if camera is None:
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret or frame is None:
            print("[video_feed] Failed to read frame from camera")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[video_feed] frame {frame_count}, mean brightness={frame.mean():.2f}")

        # Store raw frame for the detection thread to pick up
        with frame_lock:
            current_frame = frame.copy()
            # Use annotated frame (with bounding boxes) if detection has run at least once
            display_frame = annotated_frame.copy() if annotated_frame is not None else frame

        ok, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            print("[video_feed] Failed to encode frame as JPEG")
            break

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sounds/<path:filename>')
def serve_sound(filename):
    """Serve alarm WAV from ALARM_WAV_PATH when filename matches."""
    if filename == 'mixkit-space-shooter-alarm-1002.wav' and os.path.isfile(ALARM_WAV_PATH):
        return send_file(ALARM_WAV_PATH, mimetype='audio/wav')
    return send_from_directory(PROJECT_ROOT, filename)


@app.route('/video_feed')
def video_feed():
    print("[video_feed] Client connected to /video_feed")
    resp = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/api/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam and a background thread for YOLO; stream sends raw frames immediately."""
    global camera, is_detecting, video_source, fps_estimate, detection_thread
    if is_detecting:
        return jsonify({'error': 'Detection already running'}), 400

    cam_index = 0
    if platform.system() == "Windows":
        cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(cam_index)

    if not cam.isOpened():
        return jsonify({'error': 'Failed to open webcam'}), 500

    time.sleep(0.3)
    ok, test_frame = cam.read()
    if not ok or test_frame is None:
        cam.release()
        return jsonify({'error': 'Webcam opened but no frames could be read'}), 500

    camera = cam
    fps = camera.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        fps_estimate = fps
    video_source = 'webcam'
    is_detecting = True
    _start_detection_thread()

    return jsonify({'status': 'success', 'message': 'Webcam started'})


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global camera, is_detecting, video_source
    if is_detecting:
        return jsonify({'error': 'Detection already running'}), 400
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, video_file.filename)
    video_file.save(video_path)
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        return jsonify({'error': 'Failed to open video file'}), 500
    video_source = 'upload'
    is_detecting = True
    _start_detection_thread()
    return jsonify({'status': 'success', 'message': 'Video loaded'})


@app.route('/api/stop_detection', methods=['POST'])
def stop_detection():
    global camera, is_detecting, video_source
    is_detecting = False
    if camera is not None:
        camera.release()
        camera = None
    video_source = None
    with detection_lock:
        state_tracker['yawn_count'] = 0
        state_tracker['eyes_closed_frames'] = 0
        state_tracker['eyes_closed_start_time'] = None
        state_tracker['distracted_frames'] = 0
        state_tracker['phone_detected'] = False
        state_tracker['smoking_detected'] = False
        alert_states['yawn_alarm'] = False
        alert_states['yawn_alarm_start'] = None
        alert_states['eyes_closed_alarm'] = False
        alert_states['eyes_closed_alert'] = False
        alert_states['eyes_closed_email_sent'] = False
        alert_states['distracted_alert'] = False
        alert_states['phone_alert'] = False
        alert_states['phone_email_sent'] = False
        alert_states['smoking_alert'] = False
        alert_states['smoking_email_sent'] = False
    return jsonify({'status': 'success', 'message': 'Detection stopped'})


@app.route('/api/stop_yawn_alarm', methods=['POST'])
def stop_yawn_alarm():
    alert_states['yawn_alarm'] = False
    alert_states['yawn_alarm_start'] = None
    return jsonify({'status': 'success'})


@app.route('/api/status', methods=['GET'])
def get_status():
    active_thresholds = get_active_thresholds()
    with detection_lock:
        t = state_tracker['eyes_closed_start_time']
        closed_sec = round((time.time() - t) if t is not None else 0.0, 1)
        return jsonify({
            'is_detecting': is_detecting,
            'current_detection': state_tracker['last_detection'],
            'confidence': state_tracker['last_confidence'],
            'alerts': {
                'yawn_alarm': alert_states['yawn_alarm'],
                'eyes_closed_alarm': alert_states['eyes_closed_alarm'],
                'eyes_closed': alert_states['eyes_closed_alert'],
                'distracted': alert_states['distracted_alert'],
                'phone': alert_states['phone_alert'],
                'smoking': alert_states['smoking_alert']
            },
            'state': {
                'yawn_count': state_tracker['yawn_count'],
                'eyes_closed_frames': state_tracker['eyes_closed_frames'],
                'eyes_closed_seconds': closed_sec,
                'distracted_frames': state_tracker['distracted_frames']
            },
            'environment': env_context,
            'active_thresholds': active_thresholds
        })


@app.route('/api/email_config', methods=['POST'])
def set_email_config():
    data = request.json or {}
    email_config['smtp_server'] = data.get('smtp_server', 'smtp.gmail.com')
    email_config['smtp_port'] = data.get('smtp_port', 587)
    email_config['sender_email'] = data.get('sender_email', '')
    email_config['sender_password'] = data.get('sender_password', '')
    email_config['recipient_email'] = data.get('recipient_email', '')
    return jsonify({'status': 'success', 'message': 'Email configuration updated'})


@app.route('/api/mobile_config', methods=['POST'])
def set_mobile_config():
    data = request.json or {}
    mobile_config['enabled'] = bool(data.get('enabled', False))
    mobile_config['webhook_url'] = data.get('webhook_url', '').strip()
    return jsonify({'status': 'success', 'message': 'Mobile notification configuration updated'})


@app.route('/api/weather_config', methods=['POST'])
def set_weather_config():
    global last_weather_fetch_ts
    data = request.json or {}
    weather_config['enabled'] = bool(data.get('enabled', False))
    weather_config['api_key'] = data.get('api_key', '').strip()
    weather_config['city'] = data.get('city', '').strip()
    weather_config['country_code'] = data.get('country_code', 'IN').strip() or 'IN'
    weather_config['rain_sensitivity_multiplier'] = float(data.get('rain_sensitivity_multiplier', 0.8))
    weather_config['night_sensitivity_multiplier'] = float(data.get('night_sensitivity_multiplier', 0.75))
    # Force immediate refresh on next status/detection cycle.
    last_weather_fetch_ts = 0.0
    return jsonify({'status': 'success', 'message': 'Weather configuration updated'})


@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    limit = max(1, min(int(request.args.get('limit', 20)), 100))
    return jsonify({'notifications': list(notification_feed)[:limit]})


if __name__ == '__main__':
    violations_dir = os.path.join(PROJECT_ROOT, "violations")
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(violations_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
