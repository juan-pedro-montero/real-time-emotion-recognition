model_path = "best_model_finetuned_affectnet_v4.pt"

print("Loading libraries...")
import os
import cv2
import dlib
import numpy as np
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
import torch
import time
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from collections import deque

print("Libraries loaded.")
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define emotion labels and colors
emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_colors = {
    'anger': (0, 0, 255),
    'contempt': (255, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (255, 255, 0),
    'happy': (0, 255, 255),
    'sad': (255, 165, 0),
    'surprise': (255, 192, 203),
    'neutral': (255, 255, 255)
}

# Load the emotion model
emotion_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
emotion_model.classifier[1] = nn.Linear(emotion_model.classifier[1].in_features, len(emotions))
emotion_model.load_state_dict(torch.load(model_path, map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()

print("Model loaded.")
print("Starting video capture...")

# Create directory to save images
save_dir = "emotion_images"
os.makedirs(save_dir, exist_ok=True)

# Initialize video capture
video_capture = cv2.VideoCapture(0)
prev_time = time.time()
processing_times = []
prediction_times = []

last_main_emotion = None
emotion_buffer = deque(maxlen=5)  # Buffer to smooth predictions

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Predict emotion
        if emotion_model:
            start_processing = time.time()
            face_img = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            face_pil = Image.fromarray(face_rgb)
            inputs = transform(face_pil).unsqueeze(0).to(device)
            processing_times.append(time.time() - start_processing)
            start_prediction = time.time()

            with torch.no_grad():
                outputs = emotion_model(inputs)
                probs = torch.softmax(outputs, dim=1).squeeze()
                topk = torch.topk(probs, k=2)

                labels = [emotions[i.item()] for i in topk.indices]
                scores = [p.item() for p in topk.values]

            prediction_times.append(time.time() - start_prediction)

            # Update buffer
            emotion_buffer.append((labels[0], scores[0]))

            # Smooth emotion
            if len(emotion_buffer) == emotion_buffer.maxlen:
                emotion_counts = {}
                for emo, score in emotion_buffer:
                    emotion_counts[emo] = emotion_counts.get(emo, 0) + score
                smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
            else:
                smoothed_emotion = labels[0]

            # Save image if the main emotion changes
            if smoothed_emotion != last_main_emotion:
                last_main_emotion = smoothed_emotion
                timestamp = int(time.time())
                filename = f"{save_dir}/{last_main_emotion}_{timestamp}.jpg"
                cv2.imwrite(filename, face_img)

            color = emotion_colors.get(smoothed_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, smoothed_emotion, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Default case
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, "happy", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

if processing_times:
    print(f"Average image processing time: {np.mean(processing_times)*1000:.2f} ms")

if prediction_times:
    print(f"Average emotion prediction time: {np.mean(prediction_times)*1000:.2f} ms")
