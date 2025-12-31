import torch
from torchvision import transforms, models
from PIL import Image
import os

# Paths
model_path = 'model_prototype_best.pth'
class_names = ['device_use', 'hand_raised', 'looking_board', 'writing']

# Model setup
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, len(class_names))
)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Inference transform
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = infer_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()]

if __name__ == '__main__':
    test_folder = 'data/val/device_use'  # Change to any class/val folder
    for fname in os.listdir(test_folder):
        if fname.lower().endswith('.jpg'):
            path = os.path.join(test_folder, fname)
            pred = predict_image(path)
            print(f'{fname}: {pred}')
