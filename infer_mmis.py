from typing import List
import torch
from torch.utils.data import DataLoader
from model.phiseg import PHISeg
from data.mmis import MMISDataset
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
checkpoint_path = '/Users/kaiser_1/Documents/GitHub/PHi_Seg/PHISeg_7_5_checkpoint_50000.pth'
image_size = (128, 128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PHISeg(input_channels=3,
                    num_classes=2,
                    num_filters=[32, 64, 128, 192, 192, 192, 192],
                    latent_levels=5,
                    no_convs_fcomb=4,
                    beta=10.0,
                    image_size=(3, 128, 128),
                    reversible=False)
state_dict = torch.load(checkpoint_path, map_location="cpu")
# print(f"[INFO] Loaded Checkpoint Keys: {list(state_dict.keys())[:5]}")
model.load_state_dict(state_dict)
model.eval()
model.to(device)

def save_image(labels, preds, image_folder, batch, image_name):
    # labels: Tensor(b, c, w, h)
    # gts: List[Tensor(b, w, h) x n]
    # preds: List[Tensor(b, w, h) x n]
    os.makedirs(image_folder, exist_ok=True)
    for i in range(labels.shape[0]):
        image = labels[i][0].numpy()

        plt.imsave(f"{image_folder}/image_{batch}_{i}.png", image, cmap="gray")
        for id in range(len(preds)):
            pred_image = preds[id][i].cpu().numpy()
            plt.imsave(f"{image_name}.png", pred_image, cmap="gray")

def post_process(logits):
    return logits.argmax(dim=1).to(torch.float32)

img_dirs = ["/Users/kaiser_1/Documents/Data/viz_paper/Val/image*.png"]
img_paths = [img_path for img_dir in img_dirs
             for img_path in glob.glob(img_dir)]
print(len(img_paths))
if len(img_paths) == 0:
    raise ValueError("Không tìm thấy ảnh nào trong đường dẫn.")
images = []
for img_path in img_paths:
    img = Image.open(img_path)
    
    # Chuyển từ RGBA -> RGB nếu cần
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    # Chuyển thành numpy array
    img_array = np.array(img)

    transform = T.Compose([
            T.ToTensor(),
            T.Resize(image_size),
        ])

        # Áp dụng transform
    img_tensor = transform(img)
    print(f"max: {img_tensor.max()}, min: {img_tensor.min()}")
    images.append(img_tensor)

# Tạo batch (batch_size, channels, height, width)
images = torch.stack(images)
images = images.to(device)
print(f"Data shape: {images.shape}")
n_ensemble = 15  # Số lần ensemble, có thể thay đổi tùy ý
output_folder = "/Users/kaiser_1/Documents/GitHub/PHi_Seg/output_mmis1"  # Thư mục lưu kết quả


with torch.no_grad():
    for idx, img_path in enumerate(img_paths):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        preds = []

        for i in tqdm(range(n_ensemble)):
            fake_mask = torch.ones_like(images[idx:idx+1])[:, :1, ...]  # Mask cho từng ảnh
            logits = model(images[idx:idx+1], fake_mask, training=False)
            samples = model.accumulate_output(logits, use_softmax=True)
            print(f"sample_logits shape: {samples.shape}")
            print(f"sample_logits min: {samples.min()}, max: {samples.max()}")
            pred = post_process(samples.detach().cpu())
            print(pred.shape)
            preds.append(pred)

            # Lưu predict cho từng lần ensemble
            pred_image_path = os.path.join(output_folder, f"{image_name}_pred_{i}.png")
            plt.imsave(pred_image_path, pred.squeeze(0).squeeze(0).numpy(), cmap="gray")

        # Lưu ảnh gốc và predict ensemble (tổng hợp)
        save_image(images[idx:idx+1], preds, output_folder, idx, image_name)