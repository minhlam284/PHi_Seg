from typing import Literal
import glob
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from itertools import cycle

class MSMRIDataset(Dataset):

    dataset_url = None
    masks_per_image = 2

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        mask_type: Literal["ensemble", "random", "multi"] = "random",
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
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = np.load(img_path)
        print(image.shape)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        image = self.transform(image)
        image = image.type(torch.FloatTensor)
        
        mask_path = img_path.replace('image', 'label')
        mask = np.load(mask_path).astype(np.uint8)
        mask = self.transform(mask)

        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = mask[mask_id, ...]
            mask = mask.type(torch.FloatTensor).unsqueeze(0)
        elif self.mask_type == "ensemble":
            mask = mask.mean(axis=0)
            mask = mask.type(torch.FloatTensor)
        elif self.mask_type == "multi":
            mask = [mask[i, ...].type(torch.FloatTensor).unsqueeze(0) for i in range(self.masks_per_image)]
        
        return image, mask

class msmri_data:
    def __init__(self, batch_size, num_workers):
        trainset = MSMRIDataset(data_dir='/Users/kaiser_1/Documents/Data/data/mmis', train_val_test_dir="Train")
        valset = MSMRIDataset(data_dir='/Users/kaiser_1/Documents/Data/data/mmis', train_val_test_dir="Val")
        testset = MSMRIDataset(data_dir='/Users/kaiser_1/Documents/Data/data/mmis', train_val_test_dir="Val")

        self.train = cycle(DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True))
        self.val = valset
        self.test = testset

if __name__ == "__main__":
    dataset = MSMRIDataset(data_dir='/Users/kaiser_1/Documents/Data/data/mmis', mask_type="multi")
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
    
    axs[0, 0].imshow(image[:,:,0], cmap='gray')
    axs[0, 0].set_title('T1')
    axs[0, 0].axis("off")
    
    axs[0, 1].imshow(image[:,:,1], cmap='gray')
    axs[0, 1].set_title('T1C')
    axs[0, 1].axis("off")
    
    axs[0, 2].imshow(image[:,:,2], cmap='gray')
    axs[0, 2].set_title('T2')
    axs[0, 2].axis("off")
    
    sns.heatmap(mask_var, ax=axs[1, 0], cmap="viridis")
    axs[1, 0].set_title('Variance Heatmap')
    axs[1, 0].axis("off")

    axs[1, 1].imshow(mask_e, cmap='gray')
    axs[1, 1].set_title('Mask_e')
    axs[1, 1].axis("off")

    axs[1, 2].imshow(masks[:,:,0], cmap='gray')
    axs[1, 2].set_title('Mask_1')
    axs[1, 2].axis("off")
    
    axs[2, 0].imshow(masks[:,:,1], cmap='gray')
    axs[2, 0].set_title('Mask_2')
    axs[2, 0].axis("off")
    
    axs[2, 1].imshow(masks[:,:,2], cmap='gray')
    axs[2, 1].set_title('Mask_3')
    axs[2, 1].axis("off")
    
    axs[2, 2].imshow(masks[:,:,3], cmap='gray')
    axs[2, 2].set_title('Mask_4')
    axs[2, 2].axis("off")
    
    plt.tight_layout()
    plt.show()
