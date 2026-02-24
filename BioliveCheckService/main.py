from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
import tempfile
import os

app = FastAPI(title="Biological Liveness Service")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def apply_bandpass_filter(signal, fps):
    """
    WHAT HAPPENED INSIDE THIS CODE:
    1. We define the human heart rate frequency bounds. 0.75 Hz is 45 BPM, and 2.5 Hz is 150 BPM.
    2. We calculate the Nyquist frequency, which is exactly half of the video's FPS.
    3. We normalize our target frequencies against the Nyquist frequency to configure the SciPy filter.
    4. We generate a 5th-order Butterworth bandpass filter and apply it using 'filtfilt' to remove noise without shifting the signal phase.
    """
    lowcut = 0.75  # Lower bound: 45 Beats Per Minute
    highcut = 2.5  # Upper bound: 150 Beats Per Minute
    nyquist = 0.5 * fps
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Create and apply the Butterworth filter
    b, a = butter(5, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal

def process_video(video_path):
    """
    WHAT HAPPENED INSIDE THIS CODE:
    1. We open the video and verify it meets the 15 FPS minimum mathematical requirement for rPPG.
    2. We iterate through every frame, converting it to RGB for MediaPipe to map the 468 facial landmarks.
    3. We extract the forehead region (landmarks 10, 109, 338) and isolate the Green color channel, as it absorbs light best during blood volume changes.
    4. We calculate the average green intensity for that frame and append it to our raw signal array.
    5. After processing all frames, we pass the raw signal array to our SciPy bandpass filter to reveal the heartbeat wave.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps < 15.0:
        return {"status": "failed", "reason": f"FPS ({fps}) too low for rPPG"}

    raw_green_signal = []
    confidence_score = 1.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Get image dimensions to convert normalized coordinates to pixels
                h, w, _ = frame.shape
                
                # Forehead ROI coordinates
                pt1 = landmarks.landmark[10]
                pt2 = landmarks.landmark[109]
                pt3 = landmarks.landmark[338]
                
                # Approximate bounding box for the forehead
                x_min = int(min(pt1.x, pt2.x, pt3.x) * w)
                x_max = int(max(pt1.x, pt2.x, pt3.x) * w)
                y_min = int(min(pt1.y, pt2.y, pt3.y) * h)
                y_max = int(max(pt1.y, pt2.y, pt3.y) * h)
                
                # Ensure coordinates are within frame bounds
                x_min, y_min = max(0, x_min), max(0, y_min)
                
                # Extract the ROI and the Green channel (index 1)
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    green_channel = roi[:, :, 1]
                    raw_green_signal.append(np.mean(green_channel))
        else:
            confidence_score -= 0.05

    cap.release()

    # Apply math filter to extract the pulse if we have enough data
    if len(raw_green_signal) > (fps * 2):
        filtered_pulse = apply_bandpass_filter(raw_green_signal, fps)
        
        # Calculate signal variance (a strong pulse has higher variance than flat noise)
        signal_strength = np.var(filtered_pulse)
        
        if signal_strength > 0.1 and confidence_score > 0.7:
             return {"status": "passed", "liveness_score": 0.95, "confidence": confidence_score}
        else:
             return {"status": "failed", "reason": "No strong heartbeat pulse detected", "confidence": confidence_score}
    else:
        return {"status": "failed", "reason": "Not enough face frames detected", "confidence": confidence_score}

@app.post("/api/v1/liveness/verify")
async def verify(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        result = process_video(tmp_path)
    finally:
        os.remove(tmp_path)

    return result