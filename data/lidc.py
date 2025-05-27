from typing import Literal
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from itertools import cycle
from torchvision.transforms import InterpolationMode
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pickle
class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, image_size: int):
        self.dataset = dataset
        self.transform = A.Compose(transforms=[A.Resize(image_size, image_size),
                                            A.HorizontalFlip(p=0.5),
                                            ToTensorV2()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        # print(mask.shape, mask.dtype, mask.max())
        if isinstance(mask, list):
            transformed = self.transform(image=image, masks=mask)
            mask = [m.unsqueeze(0).to(torch.float32) for m in transformed["masks"]]
        else:
            transformed = self.transform(image=image, mask=mask)
            mask = transformed["mask"].unsqueeze(0).to(torch.float32)
            # print(mask.shape, mask.dtype, mask.max())
        return transformed["image"].to(torch.float32), mask
    
class LIDCDataset(Dataset):
    dataset_url = '...'
    masks_per_image = 4

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        mask_type: Literal["ensemble", "random", "multi"] = "ensemble",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.mask_type = mask_type
        self.data = {"images": [], "masks": []}

        if train_val_test_dir:
            self.load_data(f"{self.data_dir}/{train_val_test_dir}.pickle")
        else: 
            self.load_data(f"{self.data_dir}/Train.pickle")
            self.load_data(f"{self.data_dir}/Val.pickle")

    def load_data(self, datafile_path: str):
        max_bytes = 2**31 -1

        bytes_in = bytearray(0)
        input_size = os.path.getsize(datafile_path)
        with open(datafile_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        self.data["images"].extend(new_data["images"])
        self.data["masks"].extend(new_data["masks"])

    def __len__(self):
        return len(self.data["images"])
    
    def __getitem__(self, index):
        image = self.data["images"][index]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
            
        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = self.data["masks"][index][mask_id]
        
        else:
            mask = self.data["masks"][index]
            if self.mask_type == "ensemble":
                mask = np.stack(mask, axis=-1).mean(axis=-1)
                mask = (mask > 0.5).astype(np.uint8)
            elif self.mask_type == "multi":
                pass
        return image, mask

class lidc_data:
    def __init__(self, batch_size, num_workers):
        train_set = LIDCDataset(data_dir='/Users/kaiser_1/Documents/Data/data/lidc', mask_type="random", train_val_test_dir="Train")
        train_set = TransformDataset(train_set, image_size=128)

        val_set = LIDCDataset(data_dir='/Users/kaiser_1/Documents/Data/data/lidc', mask_type="multi", train_val_test_dir="Val")
        val_set = TransformDataset(val_set, image_size=128)

        test_set = LIDCDataset(data_dir='/Users/kaiser_1/Documents/Data/data/lidc', train_val_test_dir="Val")
        test_set = TransformDataset(test_set, image_size=128)

        self.train = cycle(DataLoader(train_set, batch_size=batch_size,num_workers=num_workers,shuffle=True))
        self.val = val_set
        self.test = test_set

if __name__ == "__main__":
    dataset = LIDCDataset(data_dir='/Users/kaiser_1/Documents/Data/data/lidc', mask_type="multi")
    print(len(dataset))
    id = random.randint(0, len(dataset) - 1)
    print(id)
    
    masks, image = dataset[id]
    print(image[0].shape, image[0].dtype, type(image))
    
    masks = np.stack(masks, axis=-1)
    mask_e = masks.mean(axis=-1)
    mask_var = masks.var(axis=-1)
    print(masks.shape, mask_var.shape, masks.shape)

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0, 0].imshow(image[0], cmap='gray')
    axs[0, 0].set_title('Image')
    axs[0, 0].axis("off")
    
    # axs[0, 1].imshow(mask_e, cmap='gray')
    # axs[0, 1].set_title('Mask_e')
    # axs[0, 1].axis("off")
    
    axs[0, 2].imshow(masks[:, :, 0], cmap='gray')
    axs[0, 2].set_title('Mask_0')
    axs[0, 2].axis("off")
    
    axs[1, 0].imshow(masks[:, :, 1], cmap='gray')
    axs[1, 0].set_title('Mask_1')
    axs[1, 0].axis("off")
    
    axs[1, 1].imshow(masks[:, :, 2], cmap='gray')
    axs[1, 1].set_title('Mask_2')
    axs[1, 1].axis("off")
    
    axs[1, 2].imshow(masks[:, :, 3], cmap='gray')
    axs[1, 2].set_title('Mask_3')
    axs[1, 2].axis("off")
    
    # axs[2, 0].imshow(mask_var, cmap='gray')
    # axs[2, 0].set_title('Variance')
    # axs[2, 0].axis("off")
    
    # axs[2, 1].imshow(mask_var, cmap='gray')
    # axs[2, 1].set_title('Variance')
    # axs[2, 1].axis("off")
    
    axs[2, 2] = sns.heatmap(mask_var)
    axs[2, 2].set_title('Variance')
    axs[2, 2].axis("off")
    
    plt.tight_layout()
    plt.show()