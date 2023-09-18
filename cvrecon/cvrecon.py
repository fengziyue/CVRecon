import collections

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

import torchsparse
import torchsparse.nn.functional as spf
import torch.nn.functional as F

from cvrecon import cnn2d, cnn3d, mv_fusion, utils, view_direction_encoder, SR_encoder
from cvrecon.cost_volume import ResnetMatchingEncoder, FastFeatureVolumeManager, tensor_B_to_bM, tensor_bM_to_B, TensorFormatter


class cvrecon(torch.nn.Module):
    def __init__(self, attn_heads, attn_layers, use_proj_occ, SRfeat, SR_vi_ebd, SRCV, use_cost_volume, cv_dim, cv_overall, depth_head):
        super().__init__()
        self.use_proj_occ = use_proj_occ
        self.n_attn_heads = attn_heads
        self.resolutions = collections.OrderedDict(
            [
                ["coarse", 0.16],
                ["medium", 0.08],
                ["fine", 0.04],
            ]
        )
        self.SRfeat = SRfeat
        self.SR_vi_ebd = SR_vi_ebd
        self.SRCV = SRCV
        self.use_cost_volume = use_cost_volume
        self.cv_overall = cv_overall
        self.cv_dim = cv_dim
        SRcha = [256, 128, 64]
        self.max_depth = 5.0
        self.min_depth = 0.25

        cnn2d_output_depths = [80, 40, 24]
        cnn3d_base_depths = [32, 16, 8]

        self.cnn2d = cnn2d.MnasMulti(cnn2d_output_depths, pretrained=True)
        self.upsampler = Upsampler()

        self.output_layers = torch.nn.ModuleDict()
        self.cnns3d = torch.nn.ModuleDict()
        self.view_embedders = torch.nn.ModuleDict()
        self.sr_encoder = torch.nn.ModuleDict()
        self.layer_norms = torch.nn.ModuleDict()
        self.mv_fusion = torch.nn.ModuleDict()

        if self.use_cost_volume:
            self.matching_encoder = ResnetMatchingEncoder(18, 16)  # ResNet18, CV feature dim = 16
            self.cost_volume = FastFeatureVolumeManager(
                matching_height=480 // 8,
                matching_width=640 // 8,
                num_depth_bins=64,
                mlp_channels=[202,128,128,cv_dim - 8 if cv_overall else cv_dim],
                matching_dim_size=16,
                num_source_views=8 - 1
            )
            self.cost_volume.load_state_dict(torch.load('cv02.pth'), strict=False)
            self.matching_encoder.load_state_dict(torch.load('me.pth'))

            self.tensor_formatter = TensorFormatter()

            self.cv_global_encoder = torch.nn.ModuleDict()
            self.unshared_conv = torch.nn.ModuleDict()
            for resname, cha in zip(['coarse', 'medium', 'fine'], [80, 40, 24]):
                self.cv_global_encoder[resname] = torch.nn.Sequential(
                    torch.nn.Conv2d(64+cha, cha+64, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(cha+64),
                    torch.nn.LeakyReLU(0.2, True),
                )
                self.unshared_conv[resname] = torch.nn.Conv2d((cha+7)*64, 7*64, 3, padding=1, groups=64)

           
        
        if depth_head:
            self.depth_head = torch.nn.Conv2d(48, 1, 1)
            self.depth_loss = torch.nn.L1Loss()
        else: self.depth_head = False

        prev_output_depth = 0
        for i, (resname, res) in enumerate(self.resolutions.items()):
            if self.SRfeat:
                self.sr_encoder[resname] = SR_encoder.SR_encoder(SRcha[i], cnn2d_output_depths[i]) # 1 by 1 conv to adapt SimpleRecon feat channel to required channel.
            self.view_embedders[resname] = view_direction_encoder.ViewDirectionEncoder(  # to encode camera viewing ray into 2dCNN feature
                    cnn2d_output_depths[i], L=4
                )
            self.layer_norms[resname] = torch.nn.LayerNorm(cnn2d_output_depths[i])

            if self.n_attn_heads > 0:
                self.mv_fusion[resname] = mv_fusion.MVFusionTransformer(
                    cnn2d_output_depths[i], attn_layers, self.n_attn_heads, cv_cha=self.cv_dim,
                )
            else:
                self.mv_fusion[resname] = mv_fusion.MVFusionMean()

            input_depth = prev_output_depth + cnn2d_output_depths[i]
            if i > 0:
                # additional channel for the previous level's occupancy prediction
                input_depth += 1
            conv = cnn3d.SPVCNN(
                in_channels=input_depth,
                base_depth=cnn3d_base_depths[i],
                dropout=False,
            )
            output_depth = conv.output_depth
            self.cnns3d[resname] = conv
            self.output_layers[resname] = torchsparse.nn.Conv3d(
                output_depth, 1, kernel_size=1, stride=1
            )
            prev_output_depth = conv.output_depth

    def get_img_feats(self, rgb_imgs, proj_mats, cam_positions):
        batchsize, n_imgs, _, imheight, imwidth = rgb_imgs.shape
        feats = self.cnn2d(rgb_imgs.reshape((batchsize * n_imgs, *rgb_imgs.shape[2:])))
        for resname in self.resolutions:
            f = feats[resname]
            f = self.view_embedders[resname](f, proj_mats[resname], cam_positions)
            f = f.reshape((batchsize, n_imgs, *f.shape[1:]))
            feats[resname] = f
        return feats
    
    def get_SR_feats(self, batch_SRfeats0, batch_SRfeats1, batch_SRfeats2, proj_mats, cam_positions):
        '''
        batch_SRfeats0: [4, 20, 64, 96, 128]

        return:
            SR_feats: [4, 20, 80, 30, 40], [4, 20, 40, 60, 80], [4, 20, 24, 120, 160]
        '''
        batchsize, n_imgs = batch_SRfeats0.shape[:2]
        batch_SRfeats0 = batch_SRfeats0.reshape((batchsize * n_imgs, *batch_SRfeats0.shape[2:]))  # [bs*n_imgs, c, h, w]
        batch_SRfeats1 = batch_SRfeats1.reshape((batchsize * n_imgs, *batch_SRfeats1.shape[2:]))
        batch_SRfeats2 = batch_SRfeats2.reshape((batchsize * n_imgs, *batch_SRfeats2.shape[2:]))

        batch_SRfeats0 = F.interpolate(batch_SRfeats0, [120, 160], mode='bilinear')
        batch_SRfeats1 = F.interpolate(batch_SRfeats1, [60, 80], mode='bilinear')
        batch_SRfeats2 = F.interpolate(batch_SRfeats2, [30, 40], mode='bilinear')

        feats = {}
        feats['fine'] = self.sr_encoder['fine'](batch_SRfeats0)    
        feats['medium'] = self.sr_encoder['medium'](batch_SRfeats1)
        feats['coarse'] = self.sr_encoder['coarse'](batch_SRfeats2)

        if self.SR_vi_ebd:
            feats['fine'] = self.view_embedders['fine'](feats['fine'], proj_mats['fine'], cam_positions)
            feats['medium'] = self.view_embedders['medium'](feats['medium'], proj_mats['medium'], cam_positions)
            feats['coarse'] = self.view_embedders['coarse'](feats['coarse'], proj_mats['coarse'], cam_positions)

        feats['fine'] = feats['fine'].reshape((batchsize, n_imgs, 24, 120, 160))
        feats['medium'] = feats['medium'].reshape((batchsize, n_imgs, 40, 60, 80))
        feats['coarse'] = feats['coarse'].reshape((batchsize, n_imgs, 80, 30, 40))

        return feats
    
    def compute_matching_feats(
                            self, 
                            all_frames_bm3hw
                        ):
        """ 
            Computes matching features for the current image (reference) and 
            source images.

            Unfortunately on this PyTorch branch we've noticed that the output 
            of our ResNet matching encoder is not numerically consistent when 
            batching. While this doesn't affect training (the changes are too 
            small), it does change and will affect test scores. To combat this 
            we disable batching through this module when testing and instead 
            loop through images to compute their feautures. This is stable and 
            produces exact repeatable results.

            Args:
                cur_image: image tensor of shape B3HW for the reference image.
                src_image: images tensor of shape BM3HW for the source images.
                unbatched_matching_encoder_forward: disable batching and loops 
                    through iamges to compute feaures.
            Returns:
                matching_cur_feats: tensor of matching features of size bchw for
                    the reference current image.
                matching_src_feats: tensor of matching features of size BMcHW 
                    for the source images.
        """
        if True:
            batch_size, num_views = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_encoder(f) 
                                    for f in all_frames_B3hw.split(40, dim=0)]

            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                                        matching_feats, 
                                        batch_size=batch_size, 
                                        num_views=num_views,
                                    )

        else:
            # Compute matching features and batch them to reduce variance from 
            # batchnorm when training.
            matching_feats = self.tensor_formatter(all_frames_bm3hw,
                apply_func=self.matching_encoder,
            )

        return matching_feats
    

    def construct_cv(self, batch, n_imgs):
        cvs = []
        cv_masks = []
        cur_invK = batch["cv_invK"]
        src_K = batch["cv_k"].unsqueeze(1).repeat(1, 7, 1, 1)
        min_depth = torch.tensor(self.min_depth).type_as(src_K).view(1, 1, 1, 1)
        max_depth = torch.tensor(self.max_depth).type_as(src_K).view(1, 1, 1, 1)
        matching_feats = self.compute_matching_feats(batch["rgb_imgs"])
        matching_src_feats = matching_feats[:, n_imgs:].view([-1, n_imgs, 7] + list(matching_feats.shape[2:]))
        inv_poses = batch['inv_pose'][:, n_imgs:].view([-1, n_imgs, 7, 4, 4])
        poses = batch['pose'][:, n_imgs:].view([-1, n_imgs, 7, 4, 4])

        for i in range(n_imgs):
            matching_cur_feats = matching_feats[:, i]
            matching_src_feat = matching_src_feats[:, i]

            src_cam_T_world = inv_poses[:, i]
            src_world_T_cam = poses[:, i]
            cur_cam_T_world = batch["inv_pose"][:, i, ...]
            cur_world_T_cam = batch["pose"][:, i, ...]
            with torch.cuda.amp.autocast(False):
                # Compute src_cam_T_cur_cam, a transformation for going from 3D 
                # coords in current view coordinate frame to source view coords 
                # coordinate frames.
                src_cam_T_cur_cam = src_cam_T_world @ cur_world_T_cam.unsqueeze(1)

                # Compute cur_cam_T_src_cam the opposite of src_cam_T_cur_cam. From 
                # source view to current view.
                cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(1) @ src_world_T_cam
            
            cost_volume, lowest_cost, _, overall_mask_bhw = self.cost_volume(
                                                cur_feats=matching_cur_feats,
                                                src_feats=matching_src_feat,
                                                src_extrinsics=src_cam_T_cur_cam,
                                                src_poses=cur_cam_T_src_cam,
                                                src_Ks=src_K,
                                                cur_invK=cur_invK,
                                                min_depth=min_depth,
                                                max_depth=max_depth,
                                                return_mask=True,
                                            )
            cvs.append(cost_volume.unsqueeze(1))
            cv_masks.append(overall_mask_bhw.unsqueeze(1))
        cvs = torch.cat(cvs, dim=1)  # [b, n, c, d, h, w]
        cv_masks = torch.cat(cv_masks, dim=1)
        if self.cv_overall:
            # ############################### skiped overall feat ####################################################
            # overallfeat = cvs[:, :, -1:, ::8, ...].permute(0, 1, 3, 2, 4, 5).expand([-1, -1, -1, cvs.shape[3], -1, -1])

            # ############################### conv overall feat ####################################################
            # overallfeat = cvs[:, :, -1, ...].view([-1] + list(cvs.shape[3:]))
            # overallfeat = self.cv_global_encoder(overallfeat).view(list(cvs.shape[:2]) + [8, 1, cvs.shape[-2], cvs.shape[-1]])
            # overallfeat = overallfeat.expand([-1, -1, -1, cvs.shape[3], -1, -1])

            ############################### complete overall feat ####################################################
            # overallfeat = cvs[:, :, -1:, :, ...].permute(0, 1, 3, 2, 4, 5).expand([-1, -1, -1, cvs.shape[3], -1, -1])

            # # cvs = cvs[:, :, :-1, ...]
            # cvs = torch.cat([overallfeat, cvs], dim=2)
            pass
        return cvs, cv_masks


    def forward(self, batch, voxel_inds_16):
        bs, n_imgs = batch['depth_imgs'].shape[:2]
        if self.use_cost_volume:
            cost_volume, cv_masks = self.construct_cv(batch, n_imgs)
            batch['rgb_imgs'] = batch['rgb_imgs'][:, :n_imgs]
            for b in range(bs):
                cost_volume[b][batch['cv_invalid_mask'][b].bool()] = 0

        if self.SRfeat:
            feats_2d = self.get_SR_feats(batch["SRfeat0"], batch["SRfeat1"], batch["SRfeat2"]
            , batch["proj_mats"], batch["cam_positions"])
        else:
            feats_2d = self.get_img_feats(
                batch["rgb_imgs"], batch["proj_mats"], batch["cam_positions"]
            )
        
        if not self.depth_head: depth_out = None
    
        device = voxel_inds_16.device
        proj_occ_logits = {}
        voxel_outputs = {}
        bp_data = {}
        n_subsample = {
            "medium": 2 ** 14,
            "fine": 2 ** 16,
        }

        voxel_inds = voxel_inds_16
        voxel_features = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        voxel_logits = torch.empty(
            (len(voxel_inds), 0), dtype=feats_2d["coarse"].dtype, device=device
        )
        for resname, res in self.resolutions.items():
            if self.training and resname in n_subsample:  # subsample voxels.
                # this saves memory and possibly acts as a data augmentation
                subsample_inds = get_subsample_inds(voxel_inds, n_subsample[resname])
                voxel_inds = voxel_inds[subsample_inds]
                voxel_features = voxel_features[subsample_inds]
                voxel_logits = voxel_logits[subsample_inds]

            voxel_batch_inds = voxel_inds[:, 3].long()
            voxel_coords = voxel_inds[:, :3] * res + batch["origin"][voxel_batch_inds]  # convert to unit of meters

            featheight, featwidth = feats_2d[resname].shape[-2:]

            feat_cha = {'coarse': 80, 'medium': 40, 'fine':24}
            cv_dim = self.cv_dim - 8
            if resname != 'medium':
                cur_cost_volume = F.interpolate(cost_volume.view([bs*n_imgs*cv_dim, 64, 60, 80]), [featheight, featwidth]).view([bs, n_imgs, cv_dim, 64, featheight, featwidth])
            else: cur_cost_volume = cost_volume.clone()
            feats_2d[resname] = self.cv_global_encoder[resname](torch.cat([cur_cost_volume[:,:,-1], feats_2d[resname]], dim=2).view([-1, feat_cha[resname]+64, featheight, featwidth]))
            feats_2d[resname] = feats_2d[resname].view([bs, n_imgs, feat_cha[resname]+64, featheight, featwidth])
            overallfeat = feats_2d[resname][:, :, :64].unsqueeze(3).expand([-1, -1, -1, 64, -1, -1])
            feats_2d[resname] = feats_2d[resname][:,:,64:]
            # for d in range(64):
            #     cur_cost_volume[:,:,:,d] = self.unshared_conv[resname][d](
            #                                 torch.cat([cur_cost_volume[:,:,:,d], feats_2d[resname]], dim=2).view([-1, feat_cha[resname]+7, featheight, featwidth])
            #                                 ).view([bs, n_imgs, 7, featheight, featwidth])
            cur_cost_volume = self.unshared_conv[resname](
                torch.cat([feats_2d[resname].unsqueeze(3).expand([-1,-1,-1,64,-1,-1]), cur_cost_volume], dim=2).transpose(2,3).reshape(bs*n_imgs,-1,featheight, featwidth))
            cur_cost_volume = cur_cost_volume.view([bs, n_imgs, 64, 7, featheight, featwidth]).transpose(2,3)

            cur_cost_volume = torch.cat([overallfeat, cur_cost_volume], dim=2)

            bp_uv, bp_depth, bp_mask = self.project_voxels(  # project voxels to each image plane
                voxel_coords,
                voxel_batch_inds,
                batch["proj_mats"][resname].transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data[resname] = {
                "voxel_coords": voxel_coords,
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }
            bp_feats, cur_proj_occ_logits = self.back_project_features(  # put 2dCNN features into voxels.
                bp_data[resname],
                feats_2d[resname].transpose(0, 1),
                self.mv_fusion[resname],
                cur_cost_volume if (self.SRCV or self.use_cost_volume) else None,
                cv_masks if self.use_cost_volume else None,
            )
            proj_occ_logits[resname] = cur_proj_occ_logits

            bp_feats = self.layer_norms[resname](bp_feats)

            voxel_features = torch.cat((voxel_features, bp_feats, voxel_logits), dim=-1)  # not understood !!!!
            voxel_features = torchsparse.SparseTensor(voxel_features, voxel_inds)
            try:
                voxel_features = self.cnns3d[resname](voxel_features)
            except Exception as e:
                print(e)
                return voxel_outputs, proj_occ_logits, bp_data, depth_out

            voxel_logits = self.output_layers[resname](voxel_features)
            voxel_outputs[resname] = voxel_logits

            if resname in ["coarse", "medium"]:
                # sparsify & upsample
                occupancy = voxel_logits.F.squeeze(1) > 0
                if not torch.any(occupancy):
                    return voxel_outputs, proj_occ_logits, bp_data, depth_out
                voxel_features = self.upsampler.upsample_feats(
                    voxel_features.F[occupancy]
                )
                voxel_inds = self.upsampler.upsample_inds(voxel_logits.C[occupancy])
                voxel_logits = self.upsampler.upsample_feats(voxel_logits.F[occupancy])

        return voxel_outputs, proj_occ_logits, bp_data, depth_out

    def losses(self, voxel_logits, voxel_gt, proj_occ_logits, bp_data, depth_imgs, depth_out):
        voxel_losses = {}
        proj_occ_losses = {}
        for resname in voxel_logits:
            logits = voxel_logits[resname]
            gt = voxel_gt[resname]
            cur_loss = torch.zeros(1, device=logits.F.device, dtype=torch.float32)
            if len(logits.C) > 0:
                pred_hash = spf.sphash(logits.C)
                gt_hash = spf.sphash(gt.C)
                idx_query = spf.sphashquery(pred_hash, gt_hash)
                good_query = idx_query != -1
                gt = gt.F[idx_query[good_query]]
                logits = logits.F.squeeze(1)[good_query]
                if len(logits) > 0:
                    if resname == "fine":
                        cur_loss = torch.nn.functional.l1_loss(
                            utils.log_transform(1.05 * torch.tanh(logits)),
                            utils.log_transform(gt),
                        )
                    else:
                        cur_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, gt
                        )
                    voxel_losses[resname] = cur_loss

            proj_occ_losses[resname] = compute_proj_occ_loss(
                proj_occ_logits[resname],
                depth_imgs,
                bp_data[resname],
                truncation_distance=3 * self.resolutions[resname],
            )

        loss = sum(voxel_losses.values()) + sum(proj_occ_losses.values())
        logs = {
            **{
                f"voxel_loss_{resname}": voxel_losses[resname].detach()
                for resname in voxel_losses
            },
            **{
                f"proj_occ_loss_{resname}": proj_occ_losses[resname].detach()
                for resname in proj_occ_losses
            },
        }

        if depth_out is not None:
            bs, n_imgs = depth_imgs.shape[:2]
            depth_out = F.interpolate(depth_out, [480, 640], mode="bilinear", align_corners=False,).view([bs, n_imgs, 480, 640]).float()
            mask = ((depth_imgs > 0.001) & (depth_imgs < 10))
            depth_loss = self.depth_loss(depth_out[mask], torch.log(depth_imgs)[mask])
            loss += depth_loss
            logs.update({'depth_loss': depth_loss.detach()})

        return loss, logs

    def project_voxels(
        self, voxel_coords, voxel_batch_inds, projmat, imheight, imwidth
    ):
        device = voxel_coords.device
        n_voxels = len(voxel_coords)
        n_imgs = len(projmat)
        bp_uv = torch.zeros((n_imgs, n_voxels, 2), device=device, dtype=torch.float32)
        bp_depth = torch.zeros((n_imgs, n_voxels), device=device, dtype=torch.float32)
        bp_mask = torch.zeros((n_imgs, n_voxels), device=device, dtype=torch.bool)
        batch_inds = torch.unique(voxel_batch_inds)
        for batch_ind in batch_inds:
            batch_mask = voxel_batch_inds == batch_ind
            if torch.sum(batch_mask) == 0:
                continue
            cur_voxel_coords = voxel_coords[batch_mask]

            ones = torch.ones(
                (len(cur_voxel_coords), 1), device=device, dtype=torch.float32
            )
            voxel_coords_h = torch.cat((cur_voxel_coords, ones), dim=-1)

            im_p = projmat[:, batch_ind] @ voxel_coords_h.t()
            im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
            im_x = im_x / im_z
            im_y = im_y / im_z
            im_grid = torch.stack(
                [2 * im_x / (imwidth - 1) - 1, 2 * im_y / (imheight - 1) - 1],
                dim=-1,
            )
            im_grid[torch.isinf(im_grid)] = -2
            mask = im_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

            bp_uv[:, batch_mask] = im_grid.to(bp_uv.dtype)
            bp_depth[:, batch_mask] = im_z.to(bp_uv.dtype)
            bp_mask[:, batch_mask] = mask

        return bp_uv, bp_depth, bp_mask

    def back_project_features(self, bp_data, feats, mv_fuser, SRCV=None, cv_masks=None):
        n_imgs, batch_size, in_channels, featheight, featwidth = feats.shape
        device = feats.device
        n_voxels = len(bp_data["voxel_batch_inds"])
        feature_volume_all = torch.zeros(
            n_voxels, in_channels, device=device, dtype=torch.float32
        )
        # the default proj occ prediction is true everywhere -> logits high
        proj_occ_logits = torch.full(
            (n_imgs, n_voxels), 100, device=device, dtype=feats.dtype
        )
        batch_inds = torch.unique(bp_data["voxel_batch_inds"])
        for batch_ind in batch_inds:
            batch_mask = bp_data["voxel_batch_inds"] == batch_ind
            if torch.sum(batch_mask) == 0:
                continue

            cur_bp_uv = bp_data["bp_uv"][:, batch_mask]  # [n_imgs, n_voxels, 2]
            cur_bp_depth = bp_data["bp_depth"][:, batch_mask]  # [n_imgs, n_voxels]
            cur_bp_mask = bp_data["bp_mask"][:, batch_mask]
            cur_feats = feats[:, batch_ind].view(
                n_imgs, in_channels, featheight, featwidth
            )
            
            ################# for normal ###########################
            # cur_bp_uv = cur_bp_uv.view(n_imgs, 1, -1, 2)  # [n_imgs, 1, n_voxels, 2]
            # features = torch.nn.functional.grid_sample(
            #     cur_feats,
            #     cur_bp_uv.to(cur_feats.dtype),
            #     padding_mode="reflection",
            #     align_corners=True,
            # )
            # # features [n_imgs, in_channels, 1, n_voxels]
            # features = features.view(n_imgs, in_channels, -1)  # [n_imgs, in_channels, n_voxels]
            #######################################################
            if SRCV is not None:
                cur_bp_d = ((torch.log(cur_bp_depth) -  torch.log(torch.tensor(self.min_depth))) / torch.log(torch.tensor(self.max_depth/self.min_depth))) * 2.0 - 1.0
                cur_bp_d.nan_to_num_(nan = -1)  # negative depth will cause nan in log
                cur_bp_uv3d = bp_data["bp_uv"][:, batch_mask]
                cur_bp_uvd = torch.cat([cur_bp_uv3d, cur_bp_d[...,None]], dim=-1).unsqueeze(1).unsqueeze(1)  # [n_imgs, 1, 1, n_voxels, 3]
                
                ''' cv mask before grid sample, depreciated
                cur_srcv = torch.zeros_like(SRCV[batch_ind])  # [n_imgs, ch(128), d(64), h(60), w(80)]
                cv_mask = cv_masks[batch_ind][:,None,None].expand(cur_srcv.shape)
                # cur_srcv[~cv_mask] = 100
                cur_srcv[cv_mask] = SRCV[batch_ind][cv_mask]
                '''

                
                # cv_mask = ~cv_masks[batch_ind][:,None].detach()  # invalid costs in the cost_volume
                # cv_mask = torch.nn.functional.grid_sample(
                #     cv_mask.to(cur_feats.dtype),
                #     cur_bp_uv.to(cur_feats.dtype),
                #     padding_mode="zeros",
                #     align_corners=True,
                # )  # all voxels that are polluted by the invalid costs

                cur_srcv = SRCV[batch_ind]
                features_cv = torch.nn.functional.grid_sample(
                    cur_srcv,
                    cur_bp_uvd.to(cur_srcv.dtype),
                    padding_mode="zeros",
                    align_corners=True,
                )
                # features_cv [20, C, 1, 1, 6912]
                c = features_cv.shape[1]
                features_cv = features_cv.view(n_imgs, c, -1)  # [n_imgs, C, n_voxels]

                # ################################# concat mask ################################ remember to change depth mlp channel
                # cv_mask = cv_mask.view(n_imgs, 1, -1).detach()
                # features_cv = torch.cat([cv_mask, features_cv], dim=1)
                # ###################################################################################

                # ################################# grid sample mask ################################
                # cv_mask = (cv_mask > 0).view(n_imgs, 1, -1).expand(features_cv.shape).detach()
                # features_cv[cv_mask] = 0
                # ###################################################################################

                ######################################## atten mask ################################
                #cv_mask = cv_mask.squeeze() > 0
                #cur_bp_mask[cv_mask] = False
                ####################################################################################

                # features = torch.cat([features, features_cv], dim=1)
                features = features_cv
                # features[:, -c:, :] = features_cv

            if isinstance(mv_fuser, mv_fusion.MVFusionTransformer):
                pooled_features, cur_proj_occ_logits = mv_fuser(
                    features,
                    cur_bp_depth,
                    cur_bp_mask,
                    self.use_proj_occ,
                )
                feature_volume_all[batch_mask] = pooled_features
                proj_occ_logits[:, batch_mask] = cur_proj_occ_logits
            else:
                pooled_features = mv_fuser(features.transpose(1, 2), cur_bp_mask)
                feature_volume_all[batch_mask] = pooled_features

        return (feature_volume_all, proj_occ_logits)


class Upsampler(torch.nn.Module):
    # nearest neighbor 2x upsampling for sparse 3D array

    def __init__(self):
        super().__init__()
        self.upsample_offsets = torch.nn.Parameter(
            torch.Tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 1, 0],
                    ]
                ]
            ).to(torch.int32),
            requires_grad=False,
        )
        self.upsample_mul = torch.nn.Parameter(
            torch.Tensor([[[2, 2, 2, 1]]]).to(torch.int32), requires_grad=False
        )

    def upsample_inds(self, voxel_inds):
        return (
            voxel_inds[:, None] * self.upsample_mul + self.upsample_offsets
        ).reshape(-1, 4)

    def upsample_feats(self, feats):
        return (
            feats[:, None]
            .repeat(1, 8, 1)
            .reshape(-1, feats.shape[-1])
            .to(torch.float32)
        )


def get_subsample_inds(coords, max_per_example):
    keep_inds = []
    batch_inds = coords[:, 3].unique()
    for batch_ind in batch_inds:
        batch_mask = coords[:, -1] == batch_ind
        n = torch.sum(batch_mask)
        if n > max_per_example:
            keep_inds.append(batch_mask.float().multinomial(max_per_example))
        else:
            keep_inds.append(torch.where(batch_mask)[0])
    subsample_inds = torch.cat(keep_inds).long()
    return subsample_inds


def compute_proj_occ_loss(proj_occ_logits, depth_imgs, bp_data, truncation_distance):
    batch_inds = torch.unique(bp_data["voxel_batch_inds"])
    for batch_ind in batch_inds:
        batch_mask = bp_data["voxel_batch_inds"] == batch_ind
        cur_bp_uv = bp_data["bp_uv"][:, batch_mask]
        cur_bp_depth = bp_data["bp_depth"][:, batch_mask]
        cur_bp_mask = bp_data["bp_mask"][:, batch_mask]
        cur_proj_occ_logits = proj_occ_logits[:, batch_mask]

        depth = torch.nn.functional.grid_sample(
            depth_imgs[batch_ind, :, None],
            cur_bp_uv[:, None].to(depth_imgs.dtype),
            padding_mode="zeros",
            mode="nearest",
            align_corners=False,
        )[:, 0, 0]

        proj_occ_mask = cur_bp_mask & (depth > 0)
        if torch.sum(proj_occ_mask) > 0:
            proj_occ_gt = torch.abs(cur_bp_depth - depth) < truncation_distance
            return torch.nn.functional.binary_cross_entropy_with_logits(
                cur_proj_occ_logits[proj_occ_mask],
                proj_occ_gt[proj_occ_mask].to(cur_proj_occ_logits.dtype),
            )
        else:
            return torch.zeros((), dtype=torch.float32, device=depth_imgs.device)
