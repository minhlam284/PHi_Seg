import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from train_mmis import UNetModel
import config.local_config as sys_config
from importlib.machinery import SourceFileLoader
import torch.nn.functional as F

# === Config paths ===
config_file = '/Users/kaiser_1/Documents/GitHub/PHi_Seg/infer_config.py'
checkpoint_path = '/Users/kaiser_1/Documents/GitHub/PHi_Seg/PHISeg_MMIS_checkpoint_199000.pth'
input_folder = '/Users/kaiser_1/Documents/Data/data/mmis/Val/Sample_10'
output_folder = '/Users/kaiser_1/Documents/GitHub/PHi_Seg/output'

os.makedirs(output_folder, exist_ok=True)

# === Load model ===
config_module = config_file.split('/')[-1].rstrip('.py')
exp_config = SourceFileLoader(config_module, config_file).load_module()

model = UNetModel(exp_config, logger=None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.net.to(device)
model.net.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.net.eval()

print("Model loaded successfully!")

# === Load all image-label pairs ===
image_files = sorted(glob.glob(os.path.join(input_folder, 'image_*.npy')))
label_files = sorted(glob.glob(os.path.join(input_folder, 'label_*.npy')))

assert len(image_files) == len(label_files), "Number of images and labels must match!"

def normalize_image(img):
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + 1e-8)

for img_path, lbl_path in zip(image_files, label_files):
    # Load input image
    x_b = np.load(img_path)
    if x_b.ndim == 3:
        if x_b.shape[-1] == 3:
            x_b = np.transpose(x_b, (2, 0, 1))  # HWC -> CHW
        x_tensor = torch.tensor(x_b, dtype=torch.float32).to(device)
        x_tensor = x_tensor.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    elif x_b.ndim == 2:
        x_tensor = torch.tensor(x_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    x_tensor = F.interpolate(x_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    x_tensor = normalize_image(x_tensor)
    # Load ground truth label
    label_np = np.load(lbl_path)
    if label_np.ndim == 3:
        label_np = label_np[:, :, 0]
    label_np = F.interpolate(
        torch.tensor(label_np).unsqueeze(0).unsqueeze(0).float(),
        size=(128, 128),
        mode='nearest'
    ).squeeze().numpy()

    # Inference
    with torch.no_grad():
        outputs = model.net.sampling_only(x_tensor, n_samples=5)
        softmax_outputs = model.net.accumulate_output(outputs, use_softmax=True)
        mean_softmax = softmax_outputs.mean(dim=0)
        pred_labels = torch.argmax(mean_softmax, dim=0)

    pred_labels_np = pred_labels.cpu().numpy()

    # === Save prediction ===
    base_name = os.path.basename(img_path).replace('.npy', '')
    save_path = os.path.join(output_folder, f'{base_name}_pred.npy')
    np.save(save_path, pred_labels_np)

    # === Visualization ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Input visualization
    x_np = x_b
    if x_np.ndim == 3:
        x_vis = x_np[0]
    else:
        x_vis = x_np

    axs[0].imshow(x_vis, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    # Ground truth label
    axs[1].imshow(label_np, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Ground Truth Label')
    axs[1].axis('off')

    # Prediction visualization
    axs[2].imshow(pred_labels_np, cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')

    plt.suptitle(base_name)
    plt.tight_layout()
    plt.show()

print("Done!")
