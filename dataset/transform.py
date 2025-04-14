import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset
import numpy as np
class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, image_size: int):
        self.dataset = dataset
        height, width = image_size[1], image_size[2]
        self.transform = A.Compose([
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mask, image = self.dataset[idx]

        # multi-rater mask
        if isinstance(mask, np.ndarray) and mask.ndim == 3:
            transformed = self.transform(image=np.transpose(image, (1, 2, 0)), masks=[m for m in mask])
            mask = torch.stack([m.to(torch.float32) for m in transformed["masks"]], dim=0)  # (N, H, W)

        else:  # binary mask (1, H, W)
            transformed = self.transform(image=np.transpose(image, (1, 2, 0)), mask=mask[0])
            mask = transformed["mask"].unsqueeze(0).to(torch.float32)

        image = transformed["image"].to(torch.float32)
        return mask, image  # mask trước, image sau
