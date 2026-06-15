import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av

st.set_page_config(
    page_title="Plant Disease Detection",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default WebRTC RTC configuration (STUN server) to avoid undefined name
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

display_names = {
    "Potato___Early_blight": "Early_blight",
    "Potato___Late_blight": "Late_blight",
    "Potato___healthy": "healthy"
}

disease_info = {
    "Early_blight": "Early blight is a fungal disease that causes brown circular spots on leaves and can reduce crop yield.",
    "Late_blight": "Late blight is a severe disease caused by Phytophthora infestans and can rapidly destroy potato crops.",
    "healthy": "The leaf appears healthy with no visible signs of disease."
}

model_path = "models/plant_disease_model.pth"


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


@st.cache_resource
def load_model():
    model = PlantDiseaseCNN(len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    raw_class = class_names[pred.item()]
    disease = display_names[raw_class]

    confidence = conf.item() * 100
    probs = probs.cpu().numpy()[0]

    df = pd.DataFrame({
        "Disease": [display_names[c] for c in class_names],
        "Probability": probs * 100
    })

    return disease, confidence, df


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        disease, confidence, _ = predict(image)

        text = f"{disease} {confidence:.2f}%"

        cv2.rectangle(img, (10, 10), (700, 60), (0, 255, 0), -1)
        cv2.putText(img, text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Plant Disease Detection")

k1, k2, k3 = st.columns(3)
k1.metric("Classes", "3")
k2.metric("Model", "CNN")
k3.metric("Device", device.type.upper())

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Upload Image",
    "Camera Snapshot",
    "Dataset Stats",
    "Live Webcam"
])

with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        disease, conf, df = predict(image)

        st.metric("Prediction", disease)
        st.metric("Confidence", f"{conf:.2f}%")
        st.progress(int(conf))
        st.info(disease_info[disease])

        st.bar_chart(df.set_index("Disease"))
        st.dataframe(df, use_container_width=True)

with tab2:
    cam = st.camera_input("Capture Image")

    if cam:
        image = Image.open(cam).convert("RGB")
        st.image(image, use_container_width=True)

        disease, conf, df = predict(image)

        st.metric("Prediction", disease)
        st.metric("Confidence", f"{conf:.2f}%")
        st.progress(int(conf))
        st.info(disease_info[disease])

        st.bar_chart(df.set_index("Disease"))
        st.dataframe(df, use_container_width=True)

with tab3:
    try:
        counts = np.load("dataset_distribution.npy")
    except:
        counts = np.array([1000, 1000, 1000])

    fig, ax = plt.subplots()
    ax.pie(counts, labels=[display_names[c] for c in class_names], autopct="%1.1f%%")
    st.pyplot(fig)

    df = pd.DataFrame({
        "Class": [display_names[c] for c in class_names],
        "Images": counts
    })

    st.dataframe(df)

with tab4:
    st.subheader("Live Webcam Disease Detection")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Convert BGR → RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            disease, confidence, _ = predict(image)

            text = f"{disease} ({confidence:.2f}%)"

            # Overlay text box
            cv2.rectangle(img, (10, 10), (800, 60), (0, 255, 0), -1)
            cv2.putText(
                img,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="plant-disease-webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )