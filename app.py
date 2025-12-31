import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os

st.title('Classroom Action Detection Demo')

# Class names (update if needed)
class_names = ['device_use', 'hand_raised', 'looking_board', 'writing']

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load('model_prototype_best.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image transform
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    img_tensor = infer_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        pred_class = class_names[pred.item()]
    return pred_class

# Option to upload or select from dataset
test_root = 'data/val'
all_classes = os.listdir(test_root)
all_classes = [c for c in all_classes if os.path.isdir(os.path.join(test_root, c))]

option = st.radio('Choose input method:', ['Upload an image', 'Select from test dataset'])

if option == 'Upload an image':
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        pred_class = predict_image(image)
        st.markdown(f'### Prediction: **{pred_class}**')
else:
    class_choice = st.selectbox('Select class', all_classes)
    class_dir = os.path.join(test_root, class_choice)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if images:
        img_choice = st.selectbox('Select image', images)
        img_path = os.path.join(class_dir, img_choice)
        image = Image.open(img_path).convert('RGB')
        st.image(image, caption=f'{class_choice}/{img_choice}', use_column_width=True)
        pred_class = predict_image(image)
        st.markdown(f'### Prediction: **{pred_class}**')
    else:
        st.warning('No images found in this class folder.')
