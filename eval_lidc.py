from typing import List
import torch
from torch.utils.data import DataLoader
from model.phiseg import PHISeg
from data.mmis import MMISDataset
from data.lidc import LIDCDataset
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Subset
from torch.utils.data import Dataset
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

def compute_metric(preds: List[torch.Tensor], gts: List[torch.Tensor], batch: int):
    """_summary_
    Args:
        preds (_type_): List[Tensor(b, w, h) x n]
        gts (_type_): List[Tensor(b, w, h) x m]
    """
    def cross_entropy_map(pred, target, eps=1e-6):
        """Compute per-pixel CE (no reduction)."""
        pred = pred.clamp(min=eps, max=1 - eps)
        return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    
    def normalized_cross_correlation(x, y, eps=1e-6):
        """Compute Normalized Cross Correlation between two maps."""
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sqrt(torch.sum((x - x_mean)**2) * torch.sum((y - y_mean)**2)) + eps
        return numerator / denominator
    
    def compute_iou(output: torch.Tensor, target: torch.Tensor):
        """_summary_
        
        Args:
            output (_type_): b, c, w, h
            target (_type_): b, c, w, h
        """
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have the same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ & target_).sum(axis=(-2, -1))
        union = (output_ | target_).sum(axis=(-2, -1))
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.mean()

    def compute_dice(output: torch.Tensor, target: torch.Tensor):
        """_summary_
        Args:
            output (_type_): b, c, w, h
            target (_type_): b, c, w, h
        """
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have the same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ * target_).sum(axis=(-2, -1))
        total = output_.sum(axis=(-2, -1)) + target_.sum(axis=(-2, -1))
        dice = (2. * intersection + smooth) / (total + smooth)
        
        return dice.mean()

    def compute_ged(preds: torch.Tensor, gts: torch.Tensor):
        n, m = preds.shape[0], gts.shape[0]
        # print(f"preds shape: {preds.shape}, preds type:{preds.dtype}")
        # print(f"gts shape: {gts.shape}, gts type:{gts.dtype}")
        d1, d2, d3 = 0, 0, 0
        
        for i in range(n):
            for j in range(m):
                d1 = d1 + (1 - compute_iou(preds[i], gts[j]))
        
        for i in range(n):
            for j in range(n):
                d2 = d2 + (1 - compute_iou(preds[i], preds[j]))
        
        for i in range(m):
            for j in range(m):
                d3 = d3 + (1 - compute_iou(gts[i], gts[j]))
        
        d1, d2, d3 = (2*d1)/(n*m), d2/(n*n), d3/(m*m)
        ged = d1 - d2 - d3
        return ged

    def compute_max_dice(preds: torch.Tensor, gts: torch.Tensor):
        max_dice = 0
        for gt in gts:
            dices = [compute_dice(pred, gt) for pred in preds]
            max_dice += max(dices)
        return max_dice / len(gts)

    # def compute_ncc(sample_masks: torch.Tensor,
    #                            gt_masks:     torch.Tensor,
    #                            eps:          float = 1e-8) -> torch.Tensor:
    #     """
    #     Args:
    #     sample_masks:  Tensor of shape (N, H, W), integer values per pixel
    #     gt_masks:      Tensor of shape (M, H, W),   integer values per pixel
    #     Returns:
    #     scalar Tensor: mean NCC score across the M ground-truth masks
    #     """
    #     # cast to float
    #     samples = sample_masks.to(torch.float32)  # (N, H, W)
    #     gts     = gt_masks.to(torch.float32)      # (M, H, W)
    #     # print(f"samples shape: {samples.shape}, samples type:{samples.dtype}")
    #     # print(f"gts shape: {gts.shape}, gts type:{gts.dtype}")
    #     # Compute mean sample map
    #     mean_sample = samples.mean(dim=0)         # (H, W)

    #     # Flatten everything to vectors of length H*W
    #     N, H, W = samples.shape
    #     M, H, W = gts.shape
    #     L = H * W
    #     flat_mean = mean_sample.view(L)           # (L,)
    #     flat_gts   = gts.view(M, L)               # (M, L)
    #     mm = flat_mean.mean()
    #     sm = flat_mean.std(unbiased=False) + eps
    #     A_z = (flat_mean - mm) / sm               # (L,)

    #     gm = flat_gts.mean(dim=1, keepdim=True)   # (M,1)
    #     sm2 = flat_gts.std(dim=1, unbiased=False, keepdim=True) + eps  # (M,1)
    #     G_z = (flat_gts - gm) / sm2               # (M, L)
    #     ncc_per_gt = (G_z * A_z).sum(dim=1) / L   # (M,)

    #     # return average
    #     return ncc_per_gt.mean()

    def compute_sncc(pred_samples, gt_annotations):

        mean_pred = torch.mean(pred_samples, dim=0)  # shape: (C, H, W)
        
        # Compute CE(bar_s, s) for each sample
        ce_bar_s_s = [cross_entropy_map(mean_pred, s) for s in pred_samples]
        ce_bar_s_s = torch.stack(ce_bar_s_s)  # (N_samples, C, H, W)
        mean_ce_bar_s_s = torch.mean(ce_bar_s_s, dim=0)  # (C, H, W)

        sncc_scores = []
        for y in gt_annotations:
            ce_y_s = [cross_entropy_map(y, s) for s in pred_samples]
            ce_y_s = torch.stack(ce_y_s)
            mean_ce_y_s = torch.mean(ce_y_s, dim=0)
            
            # Compute NCC between the two maps
            ncc = normalized_cross_correlation(mean_ce_bar_s_s, mean_ce_y_s)
            sncc_scores.append(ncc)

        return torch.mean(torch.tensor(sncc_scores))

    ged, ncc, max_dice, dice, iou = 0, 0, 0, 0, 0

    preds = torch.stack(preds, dim=1) # b, n, w, h
    gts = torch.stack(gts, dim=1) # b, m, 1, w, h
    gts = gts.squeeze(2) #b, m, 1, w, h -> b, m, w, h
    # print(f"preds shape: {preds.shape}, preds type:{preds.dtype}")
    # print(f"gts shape: {gts.shape}, gts type:{gts.dtype}")
    # for batch
    for _preds, _gts in zip(preds, gts):
        # _preds: n, w, h
        # _gts: m, w, h
        ged += compute_ged(_preds, _gts)
        max_dice += compute_max_dice(_preds, _gts)
        ncc += compute_sncc(_preds, _gts)

        pred = _preds.mean(dim=0) # w, h
        gt = _gts.mean(dim=0) # w, h
        dice += compute_dice(pred, gt)
        iou += compute_iou(pred, gt)
    
    batch = preds.shape[0]
    return ged/batch, ncc / batch, max_dice/batch, dice/batch, iou/batch

def save_image(labels, gts, preds, image_folder, batch):
    # labels: Tensor(b, c, w, h)
    # gts: List[Tensor(b, w, h) x n]
    # preds: List[Tensor(b, w, h) x n]
    os.makedirs(image_folder, exist_ok=True)

    for i in range(labels.shape[0]):
        image = labels[i][0].numpy()
        plt.imsave(f"{image_folder}/image_{batch}_{i}.png", image, cmap="gray")
        for id in range(len(gts)):
            gt_image = gts[id][i].cpu().numpy()
            gt_image = gt_image.squeeze(0)
            print(f"gt_image shape: {gt_image.shape}")
            plt.imsave(f"{image_folder}/gt_{batch}_{i}_{id}.png", gt_image, cmap="gray")
        for id in range(len(preds)):
            pred_image = preds[id][i].cpu().numpy()
            plt.imsave(f"{image_folder}/pred_{batch}_{i}_{id}.png", pred_image, cmap="gray")

def post_process(logits):
    return logits.argmax(dim=1).to(torch.float32)

def main(args):
    checkpoint_path = Path(args.checkpoint)

    with open(checkpoint_path.parent / f"{args.filename}.txt", "w", encoding="utf-8") as file:
        file.write(f"checkpoint: {args.checkpoint}\n")
        file.write(f"n_ensemble: {args.n_ensemble}\n")
        file.write(f"batch_size: {args.batch_size}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valset = LIDCDataset(data_dir='/Users/kaiser_1/Documents/Data/data/lidc',mask_type="multi", train_val_test_dir="Val")
    valset = TransformDataset(valset, image_size=128)
    # valset = Subset(valset, list(range(16)))
    test_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=2, shuffle=False)
    model = PHISeg(input_channels=1,
                    num_classes=2,
                    num_filters=[32, 64, 128, 192, 192, 192, 192],
                    latent_levels=5,
                    no_convs_fcomb=4,
                    beta=10.0,
                    image_size=(1, 128, 128),
                    reversible=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # print(f"[INFO] Loaded Checkpoint Keys: {list(state_dict.keys())[:5]}")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    dice, ncc, max_dice, iou, ged = {}, {}, {}, {}, {}
    for i in range(args.n_ensemble):
        dice[i], ncc[i], max_dice[i], iou[i], ged[i] = 0, 0, 0, 0, 0

    for batch_id, (patches, masks) in enumerate(tqdm(test_loader)):

        images = patches.to(device)
        if not isinstance(masks, list):
            masks = [masks]
        # print(f"image shape: {images.shape}, image type:{images.dtype}")
        # print(f"masks shape: {masks[3].shape}, masks type:{masks[3].dtype}")
        preds = []
        for i in tqdm(range(args.n_ensemble)):
            fake_mask = torch.ones_like(images)[:, :1, ...]
            logits = model(images, fake_mask, training=False)
            samples = model.accumulate_output(logits, use_softmax=True)
            preds.append(post_process(samples.detach().cpu()))
            # print(f"preds length: {len(preds)}, preds shape: {preds[0].shape}, preds type:{preds[0].dtype}")
            metrics = compute_metric(preds, masks, batch=images.shape[0])
            ged_iter, ncc_iter, max_dice_iter, dice_iter, iou_iter = metrics
            ged[i] += ged_iter
            ncc[i] += ncc_iter
            max_dice[i] += max_dice_iter
            dice[i] += dice_iter
            iou[i] += iou_iter

            with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
                file.write(f"n_ensemble: {i} --- " 
                            f"ged_iter: {ged_iter} --- "
                            f"ncc_iter: {ncc_iter} --- "
                            f"max_dice_iter: {max_dice_iter} --- "
                            f"dice_iter: {dice_iter} --- "
                            f"iou_iter: {iou_iter}\n")

        save_image(patches, masks, preds, checkpoint_path.parent / args.filename, batch_id)

    for i in range(args.n_ensemble):
        dice[i] /= len(test_loader)
        ncc[i] /=  len(test_loader)
        max_dice[i] /= len(test_loader)
        iou[i] /= len(test_loader)
        ged[i] /= len(test_loader)
        with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
            file.write(f"n_ensemble: {i} --- "
                        f"GED: {ged[i]} --- "
                        f"NCC: {ncc[i]} --- "
                        f"Max_Dice: {max_dice[i]} --- "
                        f"Dice: {dice[i]} --- "
                        f"IoU: {iou[i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data based on version")
    parser.add_argument("--checkpoint", "-ckpt", default=None, type=str, help="Specify the checkpoint")
    parser.add_argument("--batch_size", default=16, type=int, help="Override Batch size")
    parser.add_argument("--n_ensemble", default=15, type=int, help="Override number of samples to ensemble")
    parser.add_argument("--filename", default="eval", type=str, help="Specify the eval filename")
    args = parser.parse_args()
    
    main(args)