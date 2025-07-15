import torch
from torchvision import transforms
from PIL import Image
from detection.deepfake_model import DeepfakeCNN

def load_model(model_path):
    try:
        model = DeepfakeCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_image(image_path, model_path):
    model = load_model(model_path)
    if model is None:
        return None

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return None

    with torch.no_grad():
        try:
            output = model(image)
            prob = torch.sigmoid(output).item()
            print(f"✅ Prediction Score: {prob}")
            return prob if isinstance(prob, float) else None
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
