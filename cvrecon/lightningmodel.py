import os

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch

from cvrecon import collate, data, utils, cvrecon


class FineTuning(pl.callbacks.BaseFinetuning):
    def __init__(self, initial_epochs, use_cost_volume=False):
        super().__init__()
        self.initial_epochs = initial_epochs
        self.use_cost_volume = use_cost_volume

    def freeze_before_training(self, pl_module):
        modules = [
            pl_module.cvrecon.cnn2d.conv0,
            pl_module.cvrecon.cnn2d.conv1,
            pl_module.cvrecon.cnn2d.conv2,
            pl_module.cvrecon.upsampler,
        ] + ([
            pl_module.cvrecon.matching_encoder,
            pl_module.cvrecon.cost_volume.mlp.net[:4],
        ]if self.use_cost_volume else [])
        for mod in modules:
            self.freeze(mod, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch >= self.initial_epochs:
            self.unfreeze_and_add_param_group(
                modules=[
                    pl_module.cvrecon.cnn2d.conv0,
                    pl_module.cvrecon.cnn2d.conv1,
                    pl_module.cvrecon.cnn2d.conv2,
                ] + ([pl_module.cvrecon.matching_encoder,
                    pl_module.cvrecon.cost_volume.mlp.net[:4],
                    ]if self.use_cost_volume else []),
                optimizer=optimizer,
                train_bn=False,
                lr=pl_module.config["finetune_lr"],
            )
            pl_module.cvrecon.use_proj_occ = True
            for group in pl_module.optimizers().param_groups:
                group["lr"] = pl_module.config["finetune_lr"]


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cvrecon = cvrecon.cvrecon(
            config["attn_heads"], config["attn_layers"], config["use_proj_occ"], config["SRfeat"],
            config["SR_vi_ebd"], config["SRCV"], config["cost_volume"], config["cv_dim"], config["cv_overall"], config["depth_head"],
        )
        self.config = config

    def configure_optimizers(self):
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.config["initial_lr"],
        )

    # def on_train_epoch_start(self):
    #     self.epoch_train_logs = []

    def step(self, batch, batch_idx):
        voxel_coords_16 = batch["input_voxels_16"].C
        voxel_outputs, proj_occ_logits, bp_data, depth_out = self.cvrecon(batch, voxel_coords_16)
        voxel_gt = {
            "coarse": batch["voxel_gt_coarse"],
            "medium": batch["voxel_gt_medium"],
            "fine": batch["voxel_gt_fine"],
        }
        loss, logs = self.cvrecon.losses(
            voxel_outputs, voxel_gt, proj_occ_logits, bp_data, batch["depth_imgs"], depth_out
        )
        logs["loss"] = loss.detach()
        return loss, logs, voxel_outputs

    def training_step(self, batch, batch_idx):
        n_warmup_steps = 2_000
        if self.global_step < n_warmup_steps:
            target_lr = self.config["initial_lr"]
            lr = 1e-10 + self.global_step / n_warmup_steps * target_lr
            for group in self.optimizers().param_groups:
                group["lr"] = lr

        loss, logs, _ = self.step(batch, batch_idx)
        # self.epoch_train_logs.append(logs)
        for lossname, lossval in logs.items():
            self.log('train/'+lossname, lossval, on_step=True, on_epoch=True, sync_dist=True, reduce_fx='mean', rank_zero_only=True)
        return loss

    # def on_validation_epoch_start(self):
    #     self.epoch_val_logs = []

    def validation_step(self, batch, batch_idx):
        loss, logs, voxel_outputs = self.step(batch, batch_idx)
        # self.epoch_val_logs.append(logs)
        for lossname, lossval in logs.items():
            self.log('val/'+lossname, lossval, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='mean', rank_zero_only=True)
        
    def train_dataloader(self):
        return self.dataloader("train", augment=True)

    def val_dataloader(self):
        return self.dataloader("test")

    def dataloader(self, split, augment=False):
        nworkers = self.config["nworkers"]
        if split in ["val", "test"]:
            batch_size = 1
            nworkers //= 2
        elif self.current_epoch < self.config["initial_epochs"]:
            batch_size = self.config["initial_batch_size"]
        else:
            batch_size = self.config["finetune_batch_size"]

        info_files = utils.load_info_files(self.config["scannet_dir"], split)
        dset = data.Dataset(
            info_files,
            self.config["tsdf_dir"],
            self.config[f"n_imgs_{split}"],
            self.config[f"crop_size_{split}"],
            augment=augment,
            split=split,
            SRfeat=self.config["SRfeat"],
            SRCV=self.config["SRCV"],
            cost_volume=self.config["cost_volume"],
        )
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=nworkers,
            collate_fn=collate.sparse_collate_fn,
            drop_last=True,
            #persistent_workers=True,
        )


def write_mesh(outfile, logits_04):
    batch_mask = logits_04.C[:, 3] == 0
    inds = logits_04.C[batch_mask, :3].cpu().numpy()
    tsdf_logits = logits_04.F[batch_mask, 0].cpu().numpy()
    tsdf = 1.05 * np.tanh(tsdf_logits)
    tsdf_vol = utils.to_vol(inds, tsdf)

    mesh = utils.to_mesh(tsdf_vol, voxel_size=0.04, level=0, mask=~np.isnan(tsdf_vol))
    o3d.io.write_triangle_mesh(outfile, mesh)
