import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from torch import autograd
import config.local_config as sys_config
import os
import logging
import shutil
from importlib.machinery import SourceFileLoader
import argparse
import time
from medpy.metric import dc
from tqdm import tqdm
import utils
from data.mmis import mmis_data

# ensure numpy types
np.int = int
np.float = float

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class UNetModel:
    def __init__(self, exp_config, logger=None, tensorboard=True):
        # initialize network
        self.net = exp_config.model(
            input_channels=exp_config.input_channels,
            num_classes=exp_config.n_classes,
            num_filters=exp_config.filter_channels,
            latent_levels=exp_config.latent_levels,
            no_convs_fcomb=exp_config.no_convs_fcomb,
            beta=exp_config.beta,
            image_size=exp_config.image_size,
            reversible=exp_config.use_reversible
        )
        self.exp_config = exp_config
        self.batch_size = exp_config.batch_size
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', verbose=True, min_lr=1e-4, patience=50000
        )
        # load pretrained
        if exp_config.pretrained_model is not None:
            self.logger.info('Loading pretrained model %s', exp_config.pretrained_model)
            model_path = os.path.join(sys_config.project_root, 'models', exp_config.pretrained_model)
            model_name = f"{exp_config.experiment_name}_{exp_config.pretrained_model}.pth"
            log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)
            save_model_path = os.path.join(log_dir, model_name)
            if os.path.exists(save_model_path):
                self.net.load_state_dict(torch.load(save_model_path))
            else:
                self.logger.info('Pretrained file %s not found. Training from scratch.', save_model_path)
        # best metrics
        self.best_dice = -1
        self.best_loss = np.inf
        self.best_ged = np.inf
        self.best_ncc = -1

    def train(self, data):
        self.net.train()
        self.logger.info('Starting training.')
        self.logger.info('Filters: %s', self.exp_config.filter_channels)
        self.logger.info('Batch size: %d', self.batch_size)
        for self.iteration in tqdm(range(1, self.exp_config.iterations)):
            x_b, s_b = next(data.train)
            if isinstance(x_b, torch.Tensor):
                print(f"x_b type: {type(x_b)}, shape: {x_b.shape}, min: {x_b.min().item()}, max: {x_b.max().item()}, mean: {x_b.mean().item()}")
            else:
                print(f"x_b type: {type(x_b)}, shape: {x_b.shape}, min: {np.min(x_b)}, max: {np.max(x_b)}, mean: {np.mean(x_b)}")
            if isinstance(s_b, torch.Tensor):
                print(f"s_b type: {type(s_b)}, shape: {s_b.shape}, unique values: {torch.unique(s_b)}")
            else:
                print(f"s_b type: {type(s_b)}, shape: {s_b.shape}, unique values: {np.unique(s_b)}")
            print(f"s_b min: {s_b.min().item()}, max: {s_b.max().item()}, mean: {s_b.mean().item()}")
            print(f"s_b unique values (rounded): {torch.unique(s_b.round())}")
            # s_b = (s_b * 255).long()
            print(f"s_b min: {s_b.min().item()}, max: {s_b.max().item()}, unique values: {torch.unique(s_b)}")
            patch = torch.tensor(x_b, dtype=torch.float32).to(self.device)
            mask = torch.tensor(s_b, dtype=torch.float32).to(self.device)
            # forward
            self.net.forward(patch, mask, training=True)
            loss = self.net.loss(mask)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # validation and checkpoint
            if self.iteration % self.exp_config.validation_frequency == 0:
                # self.validate(data)
                self.save_model(f"checkpoint_{self.iteration}")
            # logging
            if self.iteration % self.exp_config.logging_frequency == 0:
                self.logger.info('Iteration %d Loss %.4f', self.iteration, loss.item())
            self.scheduler.step(loss)
        self.logger.info('Finished training.')

    def validate(self, data):
        self.net.eval()
        with torch.no_grad():
            self.logger.info('Validation at iteration %d', self.iteration)
            elbo_list, kl_list, recon_list = [], [], []
            dice_list, ged_list, ncc_list = [], [], []
            start = time.time()

            for x_b, s_b in data.val:
                # prepare input
                patch = torch.tensor(x_b, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_np = np.array(s_b)[0]  # use first annotator (H, W)
                mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                print(f"mask_shape:{mask.shape}, mask_dim:{mask.dim()}")

                # repeat for posterior sampling
                B = self.exp_config.validation_samples
                patch_arr = patch.repeat(B, 1, 1, 1)
                mask_arr = mask.repeat(B, 1, 1, 1)

                # posterior pass for both losses and segmentation samples
                s_list = self.net.forward(patch_arr, mask_arr, training=False)
                softmax_arr = self.net.accumulate_output(s_list, use_softmax=True)  # (B,C,H,W)

                # losses
                elbo_list.append(self.net.loss(mask_arr).item())
                kl_list.append(self.net.kl_divergence_loss.item())
                recon_list.append(self.net.reconstruction_loss.item())

                # Dice per sample
                dice_per_sample = []
                for b in range(B):
                    pred_np = softmax_arr[b].argmax(dim=0).cpu().numpy()
                    dice_vals = [dc((pred_np == lbl).astype(np.uint8), (mask_np == lbl).astype(np.uint8))
                                    for lbl in range(self.exp_config.n_classes)]
                    dice_per_sample.append(dice_vals)
                dice_arr = np.array(dice_per_sample)
                dice_list.append(dice_arr.mean(axis=0))

                # GED
                preds = softmax_arr.argmax(dim=1)  # (B,H,W)
                gts = torch.tensor(mask_np, dtype=torch.int64, device=self.device).expand(B, -1, -1)
                ged_val = utils.generalised_energy_distance(preds, gts,
                                                            nlabels=self.exp_config.n_classes - 1,
                                                            label_range=range(1, self.exp_config.n_classes))
                ged_list.append(ged_val)

                # NCC
                gt_one_hot = utils.convert_batch_to_onehot(
                    torch.tensor(mask_np, device=self.device).unsqueeze(0).unsqueeze(0),
                    nlabels=self.exp_config.n_classes
                ).to(self.device)# → shape (1, C, H, W)
                B = self.exp_config.validation_samples
                gt_one_hot = gt_one_hot.repeat(B, 1, 1, 1) # → shape (B, C, H, W)
                ncc_val = utils.variance_ncc_dist(softmax_arr, gt_one_hot)
                ncc_list.append(ncc_val)

            # aggregate metrics
            dice_per_structure = np.stack(dice_list).mean(axis=0)
            self.avg_dice = float(dice_per_structure.mean())
            self.foreground_dice = float(dice_per_structure[1])
            self.val_elbo = float(np.mean(elbo_list))
            self.val_kl_loss = float(np.mean(kl_list))
            self.val_recon_loss = float(np.mean(recon_list))
            self.avg_ged = float(np.mean(ged_list))
            self.avg_ncc = float(np.mean(ncc_list))

            self.logger.info(' - Foreground Dice: %.4f', self.foreground_dice)
            self.logger.info(' - Mean (neg.) ELBO: %.4f', self.val_elbo)
            self.logger.info(' - Mean GED: %.4f', self.avg_ged)
            self.logger.info(' - Mean NCC: %.4f', self.avg_ncc)

            # update bests
            if self.avg_dice > self.best_dice:
                self.best_dice = self.avg_dice
                self.save_model('best_dice')
            if self.val_elbo < self.best_loss:
                self.best_loss = self.val_elbo
                self.save_model('best_loss')
            if self.avg_ged < self.best_ged:
                self.best_ged = self.avg_ged
                self.save_model('best_ged')
            if self.avg_ncc > self.best_ncc:
                self.best_ncc = self.avg_ncc
                self.save_model('best_ncc')

            self.logger.info('Validation took %.2f s', time.time() - start)
        self.net.train()


    def save_model(self, savename):
        model_name = f"{self.exp_config.experiment_name}_{savename}.pth"
        log_dir = os.path.join(sys_config.log_root,
                               self.exp_config.log_dir_name,
                               self.exp_config.experiment_name)
        save_path = os.path.join(log_dir, model_name)
        torch.save(self.net.state_dict(), save_path)
        self.logger.info('Saved model: %s', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('EXP_PATH', type=str)
    parser.add_argument('LOCAL', type=str)
    parser.add_argument('dummy', type=str)
    args = parser.parse_args()
    # Load config
    config_module = args.EXP_PATH.split('/')[-1].rstrip('.py')
    if args.LOCAL == 'local':
        import config.local_config as sys_config
    exp_config = SourceFileLoader(config_module, args.EXP_PATH).load_module()
    # Setup logging & dirs
    log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)
    utils.makefolder(log_dir)
    shutil.copy(exp_config.__file__, log_dir)
    basic_logger = utils.setup_logger('basic_logger', os.path.join(log_dir, 'training_log.log'))
    # Run
    data = mmis_data(batch_size=exp_config.batch_size, num_workers=4)
    model = UNetModel(exp_config, logger=basic_logger)
    model.train(data)
    model.save_model('last')
