import torch
import cv2
import numpy as np
from model import create_unet_resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

model = create_unet_resnet18(num_classes=3)

checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)

model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Loaded model")

img = cv2.imread("test.jpg")
original = img.copy()

img = cv2.resize(img, (512,512))
img = img/255.0

mean = np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])

img = (img-mean)/std

x = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

with torch.no_grad():

    logits = model(x)

mask = logits.argmax(1).squeeze().cpu().numpy()

print("Predicted classes:", np.unique(mask))

unique, counts = np.unique(mask, return_counts=True)
print("Pixel distribution:", dict(zip(unique,counts)))

vis = np.zeros_like(mask, dtype=np.uint8)
vis[mask==1]=127
vis[mask==2]=255

cv2.imwrite("prediction_mask.png",vis)

color_mask = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)

color_mask[mask==1]=[0,255,0]
color_mask[mask==2]=[0,0,255]

color_mask=cv2.resize(color_mask,(original.shape[1],original.shape[0]))

overlay=cv2.addWeighted(original,0.7,color_mask,0.3,0)

cv2.imwrite("prediction_overlay.png",overlay)

print("Saved outputs")