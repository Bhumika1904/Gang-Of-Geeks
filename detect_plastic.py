import torch
import numpy as np
import cv2
from torchvision import transforms
from train_model import PlasticDetector

# Load the model
model = PlasticDetector()
model.load_state_dict(torch.load('models/plastic_detector.pth'))
model.eval()

# Load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(r"C:\Users\Bhumika\OneDrive\Documents\Python Scripts\pro\data\plastic\1.jpg")
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

# Detect plastic in the image and draw a boundary
def detect_plastic(image_path):
    img = preprocess_image(r"C:\Users\Bhumika\OneDrive\Documents\Python Scripts\pro\data\plastic\1.jpg")
    original_img = cv2.imread(r"C:\Users\Bhumika\OneDrive\Documents\Python Scripts\pro\data\plastic\1.jpg")
    with torch.no_grad():
        prediction = model(img)
    if prediction.item() > 0.5:
        print("Plastic detected")
        # Draw a rectangle around the detected plastic
        height, width, _ = original_img.shape
        cv2.rectangle(original_img, (0, 0), (width, height), (0, 255, 0), 2)
        cv2.putText(original_img, 'Plastic', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print("No plastic detected")
    
    # Save the image with the boundary
    cv2.imwrite('output_image.jpg', original_img)

# Test the detection
detect_plastic(r'C:\Users\Bhumika\OneDrive\Documents\Python Scripts\pro\data\plastic\1.jpg')
