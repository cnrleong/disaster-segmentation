import torch
import cv2
import numpy as np
from model import create_unet_resnet18

# ----- Load model -----
model = create_unet_resnet18()

checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])

model.eval()

# ----- Load test image -----
img_path = "test.jpg"
img = cv2.imread(img_path)

if img is None:
    raise ValueError("Could not load test.jpg. Make sure the file exists in the project folder.")

original = img.copy()

# resize to training resolution
img = cv2.resize(img, (256, 256))
img = img / 255.0

# convert to tensor
x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

# ----- Run prediction -----
with torch.no_grad():
    pred = model(x)

# convert logits → class prediction
mask = pred.argmax(1).squeeze().cpu().numpy()

# ----- Save raw mask -----
mask_vis = (mask * 120).astype(np.uint8)
cv2.imwrite("prediction.png", mask_vis)

# ----- Create overlay visualization -----
mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
mask_color = cv2.resize(mask_color, (original.shape[1], original.shape[0]))

overlay = cv2.addWeighted(original, 0.7, mask_color, 0.3, 0)

cv2.imwrite("overlay_prediction.png", overlay)

print("Prediction saved to prediction.png")
print("Overlay visualization saved to overlay_prediction.png")