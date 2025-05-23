import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import requests
import json
import io
import warnings
from transnext import transnext_base, transnext_tiny, transnext_micro  # Import transnext_base from transnext.py

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# URLs for ImageNet classes
IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
IMAGENET_CLASSES_TXT_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Hàm tải nhãn ImageNet
def load_imagenet_classes():
    global imagenet_classes
    imagenet_classes = None
    
    try:
        response = requests.get(IMAGENET_CLASS_INDEX_URL, timeout=10)
        if response.status_code == 200:
            class_index = json.loads(response.content)
            imagenet_classes = [class_index[str(i)][1] for i in range(1000)]
            print("Loaded ImageNet classes from JSON")
            return imagenet_classes
        else:
            print(f"Failed to fetch JSON labels: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error loading JSON labels: {e}")
    
    try:
        response = requests.get(IMAGENET_CLASSES_TXT_URL, timeout=10)
        if response.status_code == 200:
            imagenet_classes = response.text.strip().split('\n')
            print("Loaded ImageNet classes from TXT fallback")
            return imagenet_classes
        else:
            print(f"Failed to fetch TXT labels: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error loading TXT labels: {e}")
    
    print("Warning: Could not load ImageNet classes. Using class indices instead.")
    return [str(i) for i in range(1000)]

# Hàm tải và tiền xử lý ảnh từ URL
def preprocess_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: HTTP {response.status_code}")
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Hàm inference
def infer_image(model, image_tensor, device, labels):
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    print("\nTop-5 predictions:")
    for i in range(5):
        prob = top5_prob[0][i].item() * 100
        label_idx = top5_idx[0][i].item()
        label_name = labels[label_idx] if labels else f"Class {label_idx}"
        print(f"{i+1}. {label_name}: {prob:.2f}%")

# Main
def main():
    # Cấu hình
    model_path = 'D:/master/ChuyenDeNghienCuu/TransNeXt/classification/checkpoints/transnext_base_224_1k.pth'
    image_url = 'https://images.pexels.com/photos/128756/pexels-photo-128756.jpeg?cs=srgb&dl=pexels-crisdip-35358-128756.jpg&fm=jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tải nhãn ImageNet
    imagenet_labels = load_imagenet_classes()
    
    # Tải mô hình
    model = transnext_base(pretrained=False, img_size=224, pretrain_size=224).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict with strict=True since architecture should match
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded pretrained model from {model_path}")
    
    # Tiền xử lý ảnh
    image_tensor = preprocess_image(image_url)
    if image_tensor is None:
        print("Failed to process image. Exiting.")
        return
    
    # Chạy inference
    print("Result from TransNeXt Base:")
    infer_image(model, image_tensor, device, imagenet_labels)
    
    model_path_tiny = 'D:/master/ChuyenDeNghienCuu/TransNeXt/classification/checkpoints/transnext_tiny_224_1k.pth'
    model_tiny = transnext_tiny(pretrained=False, img_size=224, pretrain_size=224).to(device)
    checkpoint_tiny = torch.load(model_path_tiny, map_location=device)
    state_dict_tiny = checkpoint_tiny if isinstance(checkpoint_tiny, dict) else checkpoint_tiny.state_dict()
    if any(k.startswith('module.') for k in state_dict_tiny.keys()):
        state_dict_tiny = {k.replace('module.', ''): v for k, v in state_dict_tiny.items()}
    
    # Load state dict with strict=True since architecture should match
    model_tiny.load_state_dict(state_dict_tiny, strict=True)
    print(f"Loaded pretrained model from {model_path_tiny}")
    
    # Tiền xử lý ảnh
    image_tensor_tiny = preprocess_image(image_url)
    if image_tensor_tiny is None:
        print("Failed to process image. Exiting.")
        return
    
    print("Result from TransNeXt Tiny:")
    infer_image(model_tiny, image_tensor_tiny, device, imagenet_labels)

    model_path_micro = 'D:/master/ChuyenDeNghienCuu/TransNeXt/classification/checkpoints/transnext_micro_224_1k.pth'
    model_micro = transnext_micro(pretrained=False, img_size=224, pretrain_size=224).to(device)
    checkpoint_micro = torch.load(model_path_micro, map_location=device)
    state_dict_micro = checkpoint_micro if isinstance(checkpoint_micro, dict) else checkpoint_micro.state_dict()
    if any(k.startswith('module.') for k in state_dict_micro.keys()):
        state_dict_micro = {k.replace('module.', ''): v for k, v in state_dict_micro.items()}
    
    # Load state dict with strict=True since architecture should match
    model_micro.load_state_dict(state_dict_micro, strict=True)
    print(f"Loaded pretrained model from {model_path_micro}")
    
    # Tiền xử lý ảnh
    image_tensor_micro = preprocess_image(image_url)
    if image_tensor_micro is None:
        print("Failed to process image. Exiting.")
        return
    
    print("Result from TransNeXt Micro:")
    infer_image(model_micro, image_tensor_micro, device, imagenet_labels)

if __name__ == "__main__":
    main()