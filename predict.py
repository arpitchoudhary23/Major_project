import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "/home/arpit/Documents/Major_project/dataset/PlantVillage"
dataset = datasets.ImageFolder(dataset_path)
class_names = dataset.classes


class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()

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


model = PlantDiseaseCNN(len(class_names)).to(device)

model.load_state_dict(
    torch.load(
        "/home/arpit/Documents/Major_project/models/plant_disease_model.pth",
        map_location=device
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.set_page_config(page_title="Disease Detection")
st.title("Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if st.button("Predict Disease"):

    if uploaded_file is None:
        st.error("Please upload an image first")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, 1)

            disease = class_names[pred.item()]

            st.success(f"Prediction: {disease}")
            st.info(f"Confidence: {confidence.item()*100:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")