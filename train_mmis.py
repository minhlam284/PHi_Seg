import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter, FileWriter
from torch import autograd
import config.local_config as sys_config
# Python bundle packages
import os
import logging
import shutil
from importlib.machinery import SourceFileLoader
import argparse
import time
from medpy.metric import dc
from tqdm import tqdm

np.int=int
np.float=float
import utils
from data.mmis import mmis_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class UNetModel:
    def __init__(self, exp_config, logger=None, tensorboard=True):
        
        self.net = exp_config.model(input_channels = exp_config.input_channels,
                                    num_classes = exp_config.n_classes,
                                    num_filters=exp_config.filter_channels,
                                    latent_levels=exp_config.latent_levels,
                                    no_convs_fcomb=exp_config.no_convs_fcomb,
                                    beta=exp_config.beta,
                                    image_size=exp_config.image_size,
                                    reversible=exp_config.use_reversible)
        
        self.exp_config = exp_config
        self.batch_size = exp_config.batch_size
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', min_lr = 1e-4, verbose=True, patience=50000
        )
        if exp_config.pretrained_model is not None:
            self.logger.info('Loading pretrained model {}'.format(exp_config.pretrained_model))

            model_path = os.path.join(sys_config.project_root, 'models', exp_config.pretrained_model)
            model_name = self.exp_config.experiment_name + '_' + exp_config.pretrained_model + '.pth'
            log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)
            save_model_path = os.path.join(log_dir, model_name)

            if os.path.exists(model_path):
                self.net.load_state_dict(torch.load(save_model_path))
            else:
                self.logger.info('The file {} does not exist. Starting training without pretrained net.'
                                 .format(save_model_path))
        self.mean_loss_of_epoch = 0
        self.tot_loss = 0
        self.kl_loss = 0
        self.reconstruction_loss = 0
        self.dice_mean = 0
        self.val_loss = 0
        self.foreground_dice = 0

        self.val_recon_loss = 0
        self.val_elbo = 0
        self.val_kl_loss = 0
        self.avg_dice = 0
        self.avg_ged = -1
        self.avg_ncc = -1

        self.best_dice = -1
        self.best_loss = np.inf
        self.best_ged = np.inf
        self.best_ncc = -1

        if tensorboard:
            self.training_writer = SummaryWriter()
            self.validation_writer = SummaryWriter(comment='_validation')
        self.iteration = 0
    
    def train(self, data):
        self.net.train()
        self.logger.info('Starting training.')
        self.logger.info('Current filters: {}'.format(self.exp_config.filter_channels))
        self.logger.info('Batch size: {}'.format(self.batch_size))
        
        for self.iteration in tqdm(range(1, self.exp_config.iterations)):
            x_b, s_b = next(data.train) 
            print(f"x_b shape: {x_b.shape}") #(Batch, 3, H, W) -> (batch, 3, 128, 128)
            print(f"s_b shape: {s_b.shape}") #(Batch, 1, H, W) -> (batch, 1, 128, 128)
            patch = torch.tensor(x_b, dtype=torch.float32).to(self.device) #(Batch, 3, H, W) -> (batch, 3, 128, 128)
            mask = torch.tensor(s_b, dtype=torch.float32).to(self.device) #(Batch, 1, H, W) -> (batch, 1, 128, 128)
            print(f"patch shape: {patch.shape}")
            print(f"mask shape: {mask.shape}")
            self.mask = mask
            self.patch= patch
            # import IPython; IPython.embed()
            self.net.forward(patch, mask, training=True)
            self.loss = self.net.loss(mask)

            self.tot_loss += self.loss

            self.reconstruction_loss += self.net.reconstruction_loss
            self.kl_loss += self.net.kl_divergence_loss

            self.optimizer.zero_grad()

            self.loss.backward()
            self.optimizer.step()

            if self.iteration % self.exp_config.validation_frequency == 0:
                self.validate(data)
            if self.iteration % self.exp_config.logging_frequency == 0:
                self.logger.info('Iteration {} Loss {}'.format(self.iteration, self.loss))
                #self._create_tensorboard_summary()
                self.tot_loss = 0
                self.kl_loss = 0
                self.reconstruction_loss = 0

            self.scheduler.step(self.loss)

        self.logger.info('Finished training.')          

    def validate(self, data):
        self.net.eval()
        with torch.no_grad():
            self.logger.info('Validation for step {}'.format(self.iteration, self.loss))
            self.logger.info('Checkpointing model.')
            self.save_model('validation_ckpt')
            if self.device == torch.device('cuda'):
                allocated_memory = torch.cuda.max_memory_allocated(self.device)

                self.logger.info('Memory allocated in current iteration: {}{}'.format(allocated_memory, self.iteration))

            ged_list = []
            dice_list = []
            ncc_list = []
            elbo_list = []
            kl_list = []
            recon_list = []

            time_ = time.time()

            for id in range(len(data.val)):
                x_b, s_b = data.val[id] 
                print(f"x_b_val shape {x_b.shape}") #(3, 128, 128)
                print(f"s_b_val shape {s_b.shape}") #(1, 128, 128)
                patch = torch.tensor(x_b, dtype=torch.float32).to(self.device) #(3, H, W)
                val_patch = patch.unsqueeze(dim=0) #(3, H, W) -> (1, 3, H, W)                                                 

                mask = torch.tensor(s_b.squeeze(0), dtype=torch.float32).to(self.device) #(1, H, W) -> (H, W)
                val_mask = mask.unsqueeze(dim=0).unsqueeze(dim=1) #(H, W) -> (1, 1, H, W)
                val_masks = torch.tensor(mask[:,:, None], dtype=torch.float32).to(self.device) #(H, W) -> (H, W, C)
                val_masks = val_masks.transpose(0, 2).transpose(1, 2) #(H, W, C) -> (C, H, W)

                patch_arrangement = val_patch.repeat((self.exp_config.validation_samples, 1, 1, 1)) #(1, 3, H, W) -> (B, 3, H, W)
                mask_arrangement = val_mask.repeat(self.exp_config.validation_samples, 1, 1, 1) #(1, 1, H, W) -> (B, 1, H, W)

                self.mask = mask_arrangement
                self.patch = patch_arrangement

                s_out_eval_list = self.net.forward(patch_arrangement, mask_arrangement, training = False)
                s_prediction_softmax_arrangement = self.net.accumulate_outputs(s_out_eval_list, use_softmax=True)

                self.val_loss = self.net.loss(mask_arrangement)
                elbo = self.val_loss
                kl = self.net.kl_divergence_loss
                recon = self.net.reconstruction_loss

                s_prediction_softmax_mean = torch.mean(s_prediction_softmax_arrangement, axis = 0)
                s_prediction_arrangement = torch.argmax(s_prediction_softmax_arrangement, dim = 0 )

                ground_truth_arrangement = val_masks
                ged = utils.generalised_energy_distance(s_prediction_arrangement, ground_truth_arrangement,
                                                        nlabels=self.exp_config.n_classes-1,
                                                        label_range = range(1, self.exp_config.n_classes))
                
                s_gt_arr_r = val_masks.unsqueeze(dim=1) # (num_gts, nlabels, H, W)
                ground_truth_arrangement_one_hot = utils.convert_batch_to_onehot(s_gt_arr_r, nlabels=self.exp_config.n_classes)
                ncc = utils.variance_ncc_dist(s_prediction_softmax_arrangement, ground_truth_arrangement_one_hot)

                s_ = torch.argmax(s_prediction_softmax_mean, dim = 0)
                s = val_mask.view(val_mask.shape[-2], val_mask[-1])

                per_lbl_dice = []

                for lbl in range(self.exp_config.n_classes):
                    binary_pred = (s_ == lbl) * 1
                    binary_gt = (s == lbl) * 1
                    if torch.sum(binary_gt) == 0 and torch.sum(binary_pred) == 0:
                        per_lbl_dice.append(1.0)
                    elif torch.sum(binary_pred) > 0 and torch.sum(binary_gt) == 0 or torch.sum(binary_pred) == 0 and torch.sum(binary_gt) > 0:
                        per_lbl_dice.append(0.0)
                    else:
                        per_lbl_dice.append(dc(binary_pred.detach().cpu().numpy(), binary_gt.detach().cpu().numpy()))
                
                dice_list.append(per_lbl_dice)
                elbo_list.append(elbo)
                kl_list.append(kl)
                recon_list.append(recon)

                ged_list.append(ged)
                ncc_list.append(ncc)

            dice_tensor = torch.tensor(dice_list)
            per_structure_dice = dice_tensor.mean(dim=0)
            elbo_tensor = torch.tensor(elbo_list)
            kl_tensor = torch.tensor(kl_list)
            recon_tensor = torch.tensor(recon_list)

            ged_tensor = torch.tensor(ged_list)
            ncc_tensor = torch.tensor(ncc_list)

            self.avg_dice = torch.mean(dice_tensor)
            self.foreground_dice = torch.mean(dice_tensor, dim = 0)[1]
            self.val_elbo = torch.mean(elbo_tensor)
            self.val_recon_loss = torch.mean(recon_tensor)
            self.val_kl_loss = torch.mean(kl_tensor)

            self.avg_ged = torch.mean(ged_tensor)
            self.avg_ncc = torch.mean(ncc_tensor)

            self.logger.info(' - Foreground dice: %.4f' % torch.mean(self.foreground_dice))
            self.logger.info(' - Mean (neg.) ELBO: %.4f' % self.val_elbo)
            self.logger.info(' - Mean GED: %.4f' % self.avg_ged)
            self.logger.info(' - Mean NCC: %.4f' % self.avg_ncc)

            if torch.mean(per_structure_dice) >= self.best_dice:
                self.best_dice = torch.mean(per_structure_dice)
                self.logger.info('New best validation Dice! (%.3f)' % self.best_dice)
                self.save_model(savename='best_dice')
            if self.val_elbo <= self.best_loss:
                self.best_loss = self.val_elbo
                self.logger.info('New best validation loss! (%.3f)' % self.best_loss)
                self.save_model(savename='best_loss')
            if self.avg_ged <= self.best_ged:
                self.best_ged = self.avg_ged
                self.logger.info('New best GED score! (%.3f)' % self.best_ged)
                self.save_model(savename='best_ged')
            if self.avg_ncc >= self.best_ncc:
                self.best_ncc = self.avg_ncc
                self.logger.info('New best NCC score! (%.3f)' % self.best_ncc)
                self.save_model(savename='best_ncc')

            self.logger.info('Validation took {} seconds'.format(time.time()-time_))

        self.net.train()
    def test(self, data, sys_config):
        self.net.eval()
        with torch.no_grad():

            model_selection = self.exp_config.experiment_name + '_best_loss.pth'
            self.logger.info('Testing {}'.format(model_selection))

            self.logger.info('Loading pretrained model {}'.format(model_selection))

            model_path = os.path.join(
                sys_config.log_root,
                self.exp_config.log_dir_name,
                self.exp_config.experiment_name,
                model_selection)

            if os.path.exists(model_path):
                self.net.load_state_dict(torch.load(model_path))
            else:
                self.logger.info('The file {} does not exist. Aborting test function.'.format(model_path))
                return

            ged_list = []
            dice_list = []
            ncc_list = []

            time_ = time.time()

            end_dice = 0.0
            end_ged = 0.0
            end_ncc = 0.0

            for i in range(10):
                self.logger.info('Doing iteration {}'.format(i))
                n_samples = 10

                for id in range(len(data.test)):
                    x_b, s_b = data.test[id]
                    patch = torch.tensor(x_b, dtype=torch.float32).to(self.device)
                    val_patch = patch.unsqueeze(dim=0)

                    mask = torch.tensor(s_b.squeeze(0), dtype=torch.float32).to(self.device) # H, W
                    val_mask = mask.unsqueeze(dim=0).unsqueeze(dim=1) # B, C, H, W
                    val_masks = torch.tensor(mask[:,:,None], dtype=torch.float32).to(self.device)  # HWC
                    val_masks = val_masks.transpose(0, 2).transpose(1, 2)  # CHW

                    patch_arrangement = val_patch.repeat((n_samples, 1, 1, 1))

                    mask_arrangement = val_mask.repeat((n_samples, 1, 1, 1))

                    self.mask = mask_arrangement
                    self.patch = patch_arrangement

                    # training=True for constructing posterior as well
                    s_out_eval_list = self.net.forward(patch_arrangement, mask_arrangement, training=False)
                    s_prediction_softmax_arrangement = self.net.accumulate_output(s_out_eval_list, use_softmax=True)

                    s_prediction_softmax_mean = torch.mean(s_prediction_softmax_arrangement, axis=0)
                    s_prediction_arrangement = torch.argmax(s_prediction_softmax_arrangement, dim=1)

                    ground_truth_arrangement = val_masks  # nlabels, H, W
                    ged = utils.generalised_energy_distance(s_prediction_arrangement, ground_truth_arrangement,
                                                            nlabels=self.exp_config.n_classes - 1,
                                                            label_range=range(1, self.exp_config.n_classes))

                    # num_gts, nlabels, H, W
                    s_gt_arr_r = val_masks.unsqueeze(dim=1)
                    ground_truth_arrangement_one_hot = utils.convert_batch_to_onehot(s_gt_arr_r,
                                                                                     nlabels=self.exp_config.n_classes)
                    ncc = utils.variance_ncc_dist(s_prediction_softmax_arrangement, ground_truth_arrangement_one_hot)

                    s_ = torch.argmax(s_prediction_softmax_mean, dim=0)  # HW
                    s = val_mask.view(val_mask.shape[-2], val_mask.shape[-1])  # HW

                    # Write losses to list
                    per_lbl_dice = []
                    for lbl in range(self.exp_config.n_classes):
                        binary_pred = (s_ == lbl) * 1
                        binary_gt = (s == lbl) * 1

                        if torch.sum(binary_gt) == 0 and torch.sum(binary_pred) == 0:
                            per_lbl_dice.append(1.0)
                        elif torch.sum(binary_pred) > 0 and torch.sum(binary_gt) == 0 or torch.sum(
                                binary_pred) == 0 and torch.sum(
                                binary_gt) > 0:
                            per_lbl_dice.append(0.0)
                        else:
                            per_lbl_dice.append(dc(binary_pred.detach().cpu().numpy(), binary_gt.detach().cpu().numpy()))
                    dice_list.append(per_lbl_dice)

                    ged_list.append(ged)
                    ncc_list.append(ncc)

                    if ii % 100 == 0:
                        self.logger.info(' - Mean GED: %.4f' % torch.mean(torch.tensor(ged_list)))
                        self.logger.info(' - Mean NCC: %.4f' % torch.mean(torch.tensor(ncc_list)))


                dice_tensor = torch.tensor(dice_list)
                per_structure_dice = dice_tensor.mean(dim=0)

                ged_tensor = torch.tensor(ged_list)
                ncc_tensor = torch.tensor(ncc_list)

                model_path = os.path.join(
                    sys_config.log_root,
                    self.exp_config.log_dir_name,
                    self.exp_config.experiment_name)

                np.savez(os.path.join(model_path, 'ged%s_%s_2.npz' % (str(n_samples), model_selection)), ged_tensor.numpy())
                np.savez(os.path.join(model_path, 'ncc%s_%s_2.npz' % (str(n_samples), model_selection)), ncc_tensor.numpy())

                self.avg_dice = torch.mean(dice_tensor)
                self.foreground_dice = torch.mean(dice_tensor, dim=0)[1]

                self.avg_ged = torch.mean(ged_tensor)
                self.avg_ncc = torch.mean(ncc_tensor)

                logging.info('-- GED: --')
                logging.info(torch.mean(ged_tensor))
                logging.info(torch.std(ged_tensor))

                logging.info('-- NCC: --')
                logging.info(torch.mean(ncc_tensor))
                logging.info(torch.std(ncc_tensor))

                self.logger.info(' - Foreground dice: %.4f' % torch.mean(self.foreground_dice))
                self.logger.info(' - Mean (neg.) ELBO: %.4f' % self.val_elbo)
                self.logger.info(' - Mean GED: %.4f' % self.avg_ged)
                self.logger.info(' - Mean NCC: %.4f' % self.avg_ncc)

                self.logger.info('Testing took {} seconds'.format(time.time() - time_))

                end_dice += self.avg_dice
                end_ged += self.avg_ged
                end_ncc += self.avg_ncc
            self.logger.info('Mean dice: {}'.format(end_dice/10))
            self.logger.info('Mean ged: {}'.format(end_ged / 10))
            self.logger.info('Mean ncc: {}'.format(end_ncc / 10))

    def generate_images(self, data, sys_config):
        self.net.eval()
        with torch.no_grad():

            model_selection = self.exp_config.experiment_name + '_best_dice.pth'
            self.logger.info('Generating samples {}'.format(model_selection))

            self.logger.info('Loading pretrained model {}'.format(model_selection))

            model_path = os.path.join(
                sys_config.log_root,
                self.exp_config.log_dir_name,
                self.exp_config.experiment_name,
                model_selection)

            image_path = os.path.join(
                sys_config.log_root,
                self.exp_config.log_dir_name,
                self.exp_config.experiment_name,
            )

            # if os.path.exists(model_path):
            #     self.net.load_state_dict(torch.load(model_path))
            # else:
            #     self.logger.info('The file {} does not exist. Aborting test function.'.format(model_path))
            #     return

            n_samples = 10

            for ii in range(31,100):

                s_gt_arr = data.test.labels[ii, ...]

                # from HW to NCHW
                x_b = data.test.images[ii, ...]
                patch = torch.tensor(x_b, dtype=torch.float32).to(self.device)
                val_patch = patch.unsqueeze(dim=0).unsqueeze(dim=1)

                s_b = s_gt_arr[:, :, np.random.choice(self.exp_config.annotator_range)]
                mask = torch.tensor(s_b, dtype=torch.float32).to(self.device)
                val_mask = mask.unsqueeze(dim=0).unsqueeze(dim=1)
                val_masks = torch.tensor(s_gt_arr, dtype=torch.float32).to(self.device)  # HWC
                val_masks = val_masks.transpose(0, 2).transpose(1, 2)  # CHW

                patch_arrangement = val_patch.repeat((n_samples, 1, 1, 1))

                mask_arrangement = val_mask.repeat((n_samples, 1, 1, 1))

                self.mask = mask_arrangement
                self.patch = patch_arrangement

                # training=True for constructing posterior as well
                s_out_eval_list = self.net.forward(patch_arrangement, mask_arrangement, training=False)
                s_prediction_softmax_arrangement = self.net.accumulate_output(s_out_eval_list, use_softmax=True)
                s_ = torch.argmax(s_prediction_softmax_arrangement, dim=1)
                self.logger.info('s_.shape{}'.format(s_.shape))
                self.logger.info('s_'.format(s_))

                self.save_images(image_path, patch, val_masks, s_, ii)

    def save_images(self, save_location, image, ground_truth_labels, sample,
                    iteration):
        from torchvision.utils import save_image

        save_image(image, os.path.join(save_location, '{}image.png'.format(iteration)), pad_value=1, scale_each=True,
                   normalize=True)

        for i in range(self.exp_config.num_labels_per_subject):
            save_image(ground_truth_labels[i].float(),
                       os.path.join(save_location, '{}mask{}.png'.format(iteration, i)),
                       pad_value=1,
                       scale_each=True,
                       normalize=True)
        for i in range(10):
            save_image(sample[i].float(),
                       os.path.join(save_location, '{}sample{}.png'.format(iteration, i)),
                       pad_value=1,
                       scale_each=True,
                       normalize=True)


    def save_model(self, savename):
        model_name = self.exp_config.experiment_name + '_' + savename + '.pth'

        log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)
        save_model_path = os.path.join(log_dir, model_name)
        torch.save(self.net.state_dict(), save_model_path)
        self.logger.info('saved model to .pth file in {}'.format(save_model_path))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    parser.add_argument("LOCAL", type=str, help="Is this script run on the local machine or the BIWI cluster?")
    parser.add_argument("dummy", type=str, help="Is the module run with dummy training?")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')

    if args.LOCAL == 'local':
        print('Running with local configuration')
        import config.local_config as sys_config
        import matplotlib.pyplot as plt
    else:
        import config.system as sys_config

    exp_config = SourceFileLoader(config_module, config_file).load_module()

    log_dir = os.path.join(sys_config.log_root, exp_config.log_dir_name, exp_config.experiment_name)

    utils.makefolder(log_dir)

    shutil.copy(exp_config.__file__, log_dir)

    basic_logger = utils.setup_logger('basic_logger', log_dir + '/training_log.log')
    basic_logger.info('Running experiment with script: {}'.format(config_file))

    basic_logger.info('!!!! Copied exp_config file to experiment folder !!!!')

    basic_logger.info('**************************************************************')
    basic_logger.info(' *** Running Experiment: %s', exp_config.experiment_name)
    basic_logger.info('**************************************************************')

    model = UNetModel(exp_config, logger=basic_logger)
    transform = None
    #
    # trainset = bratsDataset.BratsDataset(sys_config.brats_root, exp_config, mode="train", randomCrop=None)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True,
    #                                           num_workers=1)
    #
    # model.train_brats(trainloader)

    # this loads either lidc or uzh data
    data = mmis_data(batch_size=exp_config.batch_size, num_workers=4)

    model.train(data)

    model.save_model('last')