from typing import List
import os
import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import UNetModel

from data import (
    get_lidc_dataset,
    get_mmis_dataset,
    get_qubiq_pan_dataset,
    get_qubiq_pan_les_dataset,
)

def load_experiment(path: str):
    spec = importlib.util.spec_from_file_location("expconfig", path)
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    return exp


def get_dataloader(exp):
    ds = exp.dataset.lower()
    if ds == "lidc":
        val_dataset = get_lidc_dataset(exp, mode="val")
    elif ds == "mmis":
        val_dataset = get_mmis_dataset(exp, mode="val")
    elif ds == "qubiq_pan":
        val_dataset = get_qubiq_pan_dataset(exp, mode="val")
    elif ds == "qubiq_pan_les":
        val_dataset = get_qubiq_pan_les_dataset(exp, mode="val")
    else:
        raise ValueError(f"Unknown dataset {exp.dataset}")

    print("Number of val samples:", len(val_dataset))
    return DataLoader(
        val_dataset,
        batch_size=exp.batch_size,
        num_workers=exp.num_workers,
        shuffle=True,
    )


def post_process(logits):
    return (torch.sigmoid(logits) > 0.5).to(torch.float32)


def save_image(labels, gts, preds, image_folder, batch):
    os.makedirs(image_folder, exist_ok=True)
    for i in range(labels.shape[0]):
        img = labels[i][0].cpu().numpy()
        plt.imsave(f"{image_folder}/image_{batch}_{i}.png", img, cmap="gray")
        for j, gt in enumerate(gts):
            plt.imsave(f"{image_folder}/gt_{batch}_{i}_{j}.png", gt[i].cpu().numpy(), cmap="gray")
        for j, pr in enumerate(preds):
            plt.imsave(f"{image_folder}/pred_{batch}_{i}_{j}.png", pr[i].cpu().numpy(), cmap="gray")


def compute_metric(preds: List[torch.Tensor], gts: List[torch.Tensor], batch: int):
    def compute_iou(output: torch.Tensor, target: torch.Tensor):
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        intersection = (output_ & target_).sum(axis=(-2, -1))
        union = (output_ | target_).sum(axis=(-2, -1))
        return ((intersection + smooth) / (union + smooth)).mean()

    def compute_dice(output: torch.Tensor, target: torch.Tensor):
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        intersection = (output_ * target_).sum(axis=(-2, -1))
        total = output_.sum(axis=(-2, -1)) + target_.sum(axis=(-2, -1))
        return ((2 * intersection + smooth) / (total + smooth)).mean()

    def compute_ged(preds_t: torch.Tensor, gts_t: torch.Tensor):
        n, m = preds_t.shape[0], gts_t.shape[0]
        d1 = d2 = d3 = 0.0
        for i in range(n):
            for j in range(m):
                d1 += 1 - compute_iou(preds_t[i], gts_t[j])
        for i in range(n):
            for j in range(n):
                d2 += 1 - compute_iou(preds_t[i], preds_t[j])
        for i in range(m):
            for j in range(m):
                d3 += 1 - compute_iou(gts_t[i], gts_t[j])
        d1 = 2 * d1 / (n * m)
        d2 = d2 / (n * n)
        d3 = d3 / (m * m)
        return d1 - d2 - d3

    def compute_max_dice(preds_t: torch.Tensor, gts_t: torch.Tensor):
        max_d = 0.0
        for gt in gts_t:
            dices = [compute_dice(pred, gt) for pred in preds_t]
            max_d += max(dices)
        return max_d / len(gts_t)

    def compute_ncc(sample_arr: torch.Tensor, gt_arr: torch.Tensor):
        def ncc_pair(a, v, zero_norm=True, eps=1e-8):
            a = a.flatten(); v = v.flatten()
            if zero_norm:
                a = (a - np.mean(a) + eps) / (np.std(a) * len(a) + eps)
                v = (v - np.mean(v) + eps) / (np.std(v) + eps)
            else:
                a = (a + eps) / (np.std(a) * len(a) + eps)
                v = (v + eps) / (np.std(v) + eps)
            return np.correlate(a, v).item()

        def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
            log_samples = np.log(m_samp + eps)
            return -np.sum(m_gt * log_samples, axis=0)

        sample_arr = sample_arr.unsqueeze(1).detach().cpu().numpy()
        gt_arr = gt_arr.unsqueeze(1).detach().cpu().numpy()
        mean_seg = np.mean(sample_arr, axis=0)
        N = sample_arr.shape[0]; M = gt_arr.shape[0]
        sX, sY = sample_arr.shape[2], sample_arr.shape[3]
        E_ss_arr = np.zeros((N, sX, sY))
        for i in range(N):
            E_ss_arr[i] = pixel_wise_xent(sample_arr[i], mean_seg)
        E_ss = np.mean(E_ss_arr, axis=0)
        E_sy_arr = np.zeros((M, N, sX, sY))
        for j in range(M):
            for i in range(N):
                E_sy_arr[j, i] = pixel_wise_xent(sample_arr[i], gt_arr[j])
        E_sy = np.mean(E_sy_arr, axis=1)
        ncc_list = [ncc_pair(E_ss, E_sy[j]) for j in range(M)]
        return sum(ncc_list) / M

    preds_stack = torch.stack(preds, dim=1)
    gts_stack = torch.stack(gts, dim=1)
    ged = ncc = max_dice = dice = iou = 0.0
    for _preds, _gts in zip(preds_stack, gts_stack):
        ged += compute_ged(_preds, _gts)
        max_dice += compute_max_dice(_preds, _gts)
        ncc += compute_ncc(_preds, _gts)
        pred_mean = _preds.mean(dim=0); gt_mean = _gts.mean(dim=0)
        dice += compute_dice(pred_mean, gt_mean)
        iou += compute_iou(pred_mean, gt_mean)
    b = preds_stack.shape[0]
    return ged/b, ncc/b, max_dice/b, dice/b, iou/b


def eval(exp):
    checkpoint_path = Path(exp.checkpoint)
    log_path = checkpoint_path.parent / f"{exp.filename}.txt"

    with open(log_path, "w") as f:
        f.write(f"checkpoint: {exp.checkpoint}\n")
        f.write(f"n_ensemble: {exp.n_ensemble}\n")
        f.write(f"batch_size: {exp.batch_size}\n\n")

    train_loader ,val_loader = get_dataloader(exp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetModel(exp, train_loader, val_loader)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    metrics_acc = {i: {"ged":0, "ncc":0, "max_dice":0, "dice":0, "iou":0} for i in range(exp.n_ensemble)}

    for batch_id, (masks, images) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        if not isinstance(masks, list):
            masks = [masks]
        gts = [m.squeeze(1).to(device) for m in masks]
        model(images, None, training=False)

        preds_list = []
        for i in range(exp.n_ensemble):
            logits = model.sample()
            preds = post_process(logits).squeeze(1)
            preds_list.append(preds.cpu())
            ged_i, ncc_i, max_d_i, d_i, io_u = compute_metric(preds_list, [g.cpu() for g in gts], batch=images.size(0))
            acc = metrics_acc[i]
            acc["ged"]      += ged_i
            acc["ncc"]      += ncc_i
            acc["max_dice"] += max_d_i
            acc["dice"]     += d_i
            acc["iou"]      += io_u
            with open(log_path, "a") as f:
                f.write(f"ensemble {i}: GED={ged_i:.4f}, NCC={ncc_i:.4f}, MaxDice={max_d_i:.4f}, Dice={d_i:.4f}, IoU={io_u:.4f}\n")
        save_image(images.cpu(), [g.cpu() for g in gts], preds_list, checkpoint_path.parent/exp.filename, batch_id)

    N = len(val_loader)
    with open(log_path, "a") as f:
        f.write("\n=== Average metrics ===\n")
        for i, acc in metrics_acc.items():
            f.write(f"ensemble {i} avg: GED={acc['ged']/N:.4f}, NCC={acc['ncc']/N:.4f}, MaxDice={acc['max_dice']/N:.4f}, Dice={acc['dice']/N:.4f}, IoU={acc['iou']/N:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment.py")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint")
    args = parser.parse_args()

    exp = load_experiment(args.experiment)
    if args.checkpoint:
        exp.checkpoint = args.checkpoint

    eval(exp)
