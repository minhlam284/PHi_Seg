from typing import Literal
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from itertools import cycle
from torchvision.transforms import InterpolationMode

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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

class MMISDataset(Dataset):

    dataset_url = "https://mmis2024.com/info?task=1"
    masks_per_image = 4

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        mask_type: Literal["ensemble", "random", "multi"] = "ensemble",
    ) -> None:
        super().__init__()

        if train_val_test_dir:
            img_dirs = [f"{data_dir}/{train_val_test_dir}/*/image*.npy"]
        else:
            img_dirs = [
                f"{data_dir}/Train/*/image*.npy",
                f"{data_dir}/Val/*/image*.npy",
            ]

        self.img_paths = [
            img_path for img_dir in img_dirs
            for img_path in glob.glob(img_dir)
        ]
        self.mask_type = mask_type

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        image = np.load(img_path)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        # print(image.shape)
        mask_path = img_path.replace('image', 'label')
        mask = np.load(mask_path).astype(np.uint8)
        
        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = mask[:, :, mask_id]
        elif self.mask_type == "ensemble":
            mask = mask.mean(axis=-1)
            mask = (mask > 0.5).astype(np.uint8)
        elif self.mask_type == "multi":
            mask = [mask[:,:, i] for i in range(self.masks_per_image)]
        
        return image, mask


class mmis_data:
    def __init__(self, batch_size, num_workers):
        trainset = MMISDataset(data_dir='/Users/kaiser_1/Documents/Data/mmis_test',mask_type="random", train_val_test_dir="Train")
        trainset = TransformDataset(trainset, image_size=128)
        valset = MMISDataset(data_dir='/Users/kaiser_1/Documents/Data/mmis_test',mask_type="multi", train_val_test_dir="Val")
        valset = TransformDataset(valset, image_size=128)
        testset = MMISDataset(data_dir='/Users/kaiser_1/Documents/Data/mmis_test', train_val_test_dir="Val")
        testset = TransformDataset(testset, image_size=128)

        self.train = cycle(DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True))
        self.val = valset
        self.test = testset

if __name__ == "__main__":
    dataset = MMISDataset(data_dir='/Users/kaiser_1/Documents/Data/data/mmis', mask_type="multi")
    print(len(dataset))

    id = random.randint(0, len(dataset) - 1)
    print(id)
    image, masks = dataset[id]
    print(image.shape, image.dtype, type(image))
    
    masks = np.stack(masks, axis=0)
    mask_e = masks.mean(axis=0)
    mask_var = masks.var(axis=0)

    print(mask_var.shape, masks.shape)

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    
    axs[0, 0].imshow(image[0,:,:], cmap='gray')
    axs[0, 0].set_title('T1')
    axs[0, 0].axis("off")
    
    axs[0, 1].imshow(image[1,:,:], cmap='gray')
    axs[0, 1].set_title('T1C')
    axs[0, 1].axis("off")
    
    axs[0, 2].imshow(image[2,:,:], cmap='gray')
    axs[0, 2].set_title('T2')
    axs[0, 2].axis("off")

    mask_var_2d = mask_var[0]
    axs[1, 0].imshow(mask_var_2d, cmap='viridis')
    axs[1, 0].set_title('Variance Heatmap')
    axs[1, 0].axis('off')

    mask_e_2d = mask_e[0]
    axs[1, 1].imshow(mask_e_2d, cmap='gray')
    axs[1, 1].set_title('Mask_e')
    axs[1, 1].axis("off")

    mask0 = masks[0, 0, :, :]        # shape (128, 128)
    mask1 = masks[1, 0, :, :]
    mask2 = masks[2, 0, :, :]
    mask3 = masks[3, 0, :, :]  
    axs[1, 2].imshow(mask0, cmap='gray')
    axs[1, 2].set_title('Mask_1')
    axs[1, 2].axis("off")
    
    axs[2, 0].imshow(mask1, cmap='gray')
    axs[2, 0].set_title('Mask_2')
    axs[2, 0].axis("off")
    
    axs[2, 1].imshow(mask2, cmap='gray')
    axs[2, 1].set_title('Mask_3')
    axs[2, 1].axis("off")
    
    axs[2, 2].imshow(mask3, cmap='gray')
    axs[2, 2].set_title('Mask_4')
    axs[2, 2].axis("off")
    
    plt.tight_layout()
    plt.show()



