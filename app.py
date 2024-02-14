import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import tempfile
import base64

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# setup
emotions = {
    0: ['Tidak Percaya Diri', (255,0,0), (255,255,255)],
    1: ['Tidak Percaya Diri', (255,0,0), (255,255,255)],
    2: ['Gugup', (255,255,0), (0,51,51)],
    3: ['Percaya Diri', (0,255,0), (255,255,255)],
    4: ['Gugup', (255,255,0), (0,51,51)],
    5: ['Percaya Diri', (0,255,0), (255,255,255)],
    6: ['Netral', (0,0,255), (255,255,255)]
}
num_classes = len(emotions)
input_shape = (48, 48, 1)
weights_1 = 'saved_models/vggnet.h5'
weights_2 = 'saved_models/vggnet_up.h5'

class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization(name='batch_normalization_1'))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_2'))
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_3'))
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_4'))
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_5'))
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_6'))
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_7'))
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization(name='batch_normalization_8'))
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        
        self.checkpoint_path = checkpoint_path

model_1 = VGGNet(input_shape, num_classes, weights_1)
model_1.load_weights(model_1.checkpoint_path)

model_2 = VGGNet(input_shape, num_classes, weights_2)
model_2.load_weights(model_2.checkpoint_path)

# inference
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def resize_face(face):
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

def inference(image):
    H, W, _ = image.shape
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    labels = []
    pos_faces = []

    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)

            face = image[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            pos_faces.append((x1, y1, x2, y2))

            faces = recognition_preprocessing([face])
            y_1 = model_1.predict(faces)
            y_2 = model_2.predict(faces)
            l = np.argmax(y_1 + y_2, axis=1)
            labels.append(l[0])

    return labels, pos_faces

st.title('Prediksi Ekspresi Video Hasil Interview')

file_path = st.file_uploader("Upload a video", type=["mp4", "avi"])

if file_path:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_path.read())
        file_name = tmp_file.name
    
    cap = cv2.VideoCapture(file_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_h = 360
    target_w = int(target_h * frame_width / frame_height)

    frames = []
    predictions = []
    result_frames = []
    pos_faces_list = []
    
    with st.spinner('Processing video...'):
        while True:
            success, image = cap.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (target_w, target_h))
                frames.append(image)
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
    os.unlink(file_name)

    with st.spinner('Predicting...'):
        for frame in frames:
            l, pos = inference(frame)
            predictions.append(l)
            pos_faces_list.append(pos)
            result = np.zeros_like(frame)
            for i in range(len(l)):
                cv2.rectangle(result, (pos[i][0], pos[i][1]),
                                (pos[i][2], pos[i][3]), emotions[l[i]][1], 2, lineType=cv2.LINE_AA)
                
                cv2.rectangle(result, (pos[i][0], pos[i][1]-20),
                                (pos[i][2]+20, pos[i][1]), emotions[l[i]][1], -1, lineType=cv2.LINE_AA)
                
                cv2.putText(result, f'{emotions[l[i]][0]}', (pos[i][0], pos[i][1]-5),
                                0, 0.6, emotions[l[i]][2], 2, lineType=cv2.LINE_AA)
            result_frames.append(result)

    # Buat video hasil
    output_path = "output_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))

    # Initialize counters for each emotion label
    emotion_counts = {emotion[0]: 0 for emotion in emotions.values()}

    for frame, prediction, pos_faces in zip(frames, predictions, pos_faces_list):
        for label in prediction:
            emotion_counts[emotions[label][0]] += 1
            for pos in pos_faces:
                cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), emotions[label][1], 2, lineType=cv2.LINE_AA)
                cv2.rectangle(frame, (pos[0], pos[1]-20), (pos[2]+20, pos[1]), emotions[label][1], -1, lineType=cv2.LINE_AA)
                cv2.putText(frame, f'{emotions[label][0]}', (pos[0], pos[1]-5), 0, 0.6, emotions[label][2], 2, lineType=cv2.LINE_AA)
        out.write(frame)

    out.release()

    st.title("Hasil Prediksi")
    col1, col2 = st.columns(2)

    with col1:
        st.title("Input Video")
        st.video(file_path)

    with col2:
        st.title("Hasil Prediksi Video")
        st.video(output_path, format='video/mp4', start_time=0)

    st.write("Emotion Prediksi Percapture:")
    for emotion, count in emotion_counts.items():
        st.write(f"{emotion}: {count}")

    st.write("Kesimpulan:")
    # Ambil label yang paling banyak muncul
    max_emotion = max(emotion_counts, key=emotion_counts.get)
    if max_emotion == 'Percaya Diri':
        st.success("Masuk")
    elif max_emotion == 'Netral':
        st.warning("Dipertimbangkan, Kemungkinan diterima")
    else:
        st.error("Ditolak")

    # Download video
    with st.spinner('Downloading result video...'):
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.markdown(f'<a href="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" download="{file_path.name}">Download Result Video</a>', unsafe_allow_html=True)
