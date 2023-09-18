import argparse
import json
import os
import yaml

import imageio
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import skimage.measure
import torch
import torchsparse
import tqdm
import torch.nn.functional as F

from cvrecon import data, lightningmodel, utils


import matplotlib.pyplot as plt
from PIL import Image
def colormap_image(
                image_1hw,
                mask_1hw=None, 
                invalid_color=(0.0, 0, 0.0), 
                flip=True,
                vmin=None,
                vmax=None, 
                return_vminvmax=False,
                colormap="turbo",
            ):
    """
    Colormaps a one channel tensor using a matplotlib colormap.

    Args: 
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels. 
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the 
            tensor.
        return_vminvmax: when true, returns vmin and vmax.

    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.


    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(
                            plt.cm.get_cmap(colormap)(
                                                torch.linspace(0, 1, 256)
                                            )[:, :3]
                        ).to(image_1hw.device)
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)
                                        ].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw


def save_gif(rgb_imgfiles, output_path):
    gif = []
    for fname in rgb_imgfiles:
        gif.append(Image.open(fname))
    gif[0].save(os.path.join(output_path, 'rgb.gif'), save_all=True,optimize=False, append_images=gif[1:], loop=0)


def load_model(ckpt_file, use_proj_occ, config):
    model = lightningmodel.LightningModel.load_from_checkpoint(
        ckpt_file,
        config=config,
    )
    model.cvrecon.use_proj_occ = use_proj_occ
    model = model.cuda()
    model = model.eval()
    model.requires_grad_(False)
    return model


def load_scene(info_file):
    with open(info_file, "r") as f:
        info = json.load(f)

    rgb_imgfiles = [frame["filename_color"] for frame in info["frames"]]
    depth_imgfiles = [frame["filename_depth"] for frame in info["frames"]]
    pose = np.empty((len(info["frames"]), 4, 4), dtype=np.float32)
    for i, frame in enumerate(info["frames"]):
        pose[i] = frame["pose"]
    intr = np.array(info["intrinsics"], dtype=np.float32)
    return rgb_imgfiles, depth_imgfiles, pose, intr


def get_scene_bounds(pose, intr, imheight, imwidth, frustum_depth):
    frust_pts_img = np.array(
        [
            [0, 0],
            [imwidth, 0],
            [imwidth, imheight],
            [0, imheight],
        ]
    )
    frust_pts_cam = (
        np.linalg.inv(intr) @ np.c_[frust_pts_img, np.ones(len(frust_pts_img))].T
    ).T * frustum_depth
    frust_pts_world = (
        pose @ np.c_[frust_pts_cam, np.ones(len(frust_pts_cam))].T
    ).transpose(0, 2, 1)[..., :3]

    minbound = np.min(frust_pts_world, axis=(0, 1))
    maxbound = np.max(frust_pts_world, axis=(0, 1))
    return minbound, maxbound


def get_tiles(minbound, maxbound, cropsize_voxels_fine, voxel_size_fine):
    cropsize_m = cropsize_voxels_fine * voxel_size_fine

    assert np.all(cropsize_voxels_fine % 4 == 0)
    cropsize_voxels_coarse = cropsize_voxels_fine // 4
    voxel_size_coarse = voxel_size_fine * 4

    ncrops = np.ceil((maxbound - minbound) / cropsize_m).astype(int)
    x = np.arange(ncrops[0], dtype=np.int32) * cropsize_voxels_coarse[0]
    y = np.arange(ncrops[1], dtype=np.int32) * cropsize_voxels_coarse[1]
    z = np.arange(ncrops[2], dtype=np.int32) * cropsize_voxels_coarse[2]
    yy, xx, zz = np.meshgrid(y, x, z)
    tile_origin_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)
    y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)
    z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)
    yy, xx, zz = np.meshgrid(y, x, z)
    base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]

    tiles = []
    for origin_ind in tile_origin_inds:
        origin = origin_ind * voxel_size_coarse + minbound
        tile = {
            "origin_ind": origin_ind,
            "origin": origin.astype(np.float32),
            "maxbound_ind": origin_ind + cropsize_voxels_coarse,
            "voxel_inds": torch.from_numpy(base_voxel_inds + origin_ind),
            "voxel_coords": torch.from_numpy(
                base_voxel_inds * voxel_size_coarse + origin
            ).float(),
            "voxel_features": torch.empty(
                (len(base_voxel_inds), 0), dtype=torch.float32
            ),
            "voxel_logits": torch.empty((len(base_voxel_inds), 0), dtype=torch.float32),
        }
        tiles.append(tile)
    return tiles


def frame_selection(tiles, pose, intr, imheight, imwidth, n_imgs, rmin_deg, tmin, SRlist, rgb_imgfiles, CVDict):
    sparsified_frame_inds = np.array(utils.remove_redundant(pose, rmin_deg, tmin))

    if SRlist is not None:
        SRlist_inds = []
        for frame_ind in sparsified_frame_inds:
            if '0' + rgb_imgfiles[frame_ind][-9:-4] in SRlist:
                SRlist_inds.append(frame_ind)
        if len(sparsified_frame_inds) != len(SRlist_inds):
            print('!!!!!!!!!!!!!!!!!', len(sparsified_frame_inds), len(SRlist_inds), scene_name)
            sparsified_frame_inds = np.array(SRlist_inds)
    
    if CVDict is not None:
        SRlist_inds = []
        for frame_ind in sparsified_frame_inds:
            if '0' + rgb_imgfiles[frame_ind][-9:-4] in CVDict:
                SRlist_inds.append(frame_ind)
        if len(sparsified_frame_inds) != len(SRlist_inds):
            print('!!!!!!!!!!!!!!!!!', len(sparsified_frame_inds), len(SRlist_inds), scene_name)
            sparsified_frame_inds = np.array(SRlist_inds)

    if len(sparsified_frame_inds) < n_imgs:
        print('@@@@@@@@@@@@@@@@@', scene_name, len(sparsified_frame_inds))
        # after redundant frame removal we can end up with too few frames--
        # add some back in
        avail_inds = list(set(np.arange(len(pose))) - set(sparsified_frame_inds))
        n_needed = n_imgs - len(sparsified_frame_inds)
        extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)
        selected_frame_inds = np.concatenate((sparsified_frame_inds, extra_inds))
    else:
        selected_frame_inds = sparsified_frame_inds

    for i, tile in enumerate(tiles):
        if len(selected_frame_inds) > n_imgs:
            sample_pts = tile["voxel_coords"].numpy()
            cur_frame_inds, score = utils.frame_selection(
                pose[selected_frame_inds],
                intr,
                imwidth,
                imheight,
                sample_pts,
                tmin,
                rmin_deg,
                n_imgs,
            )
            tile["frame_inds"] = selected_frame_inds[cur_frame_inds]
        else:
            tile["frame_inds"] = selected_frame_inds
    return tiles


def get_img_feats(cvrecon, imheight, imwidth, proj_mats, rgb_imgfiles, cam_positions, SRfeat, scene_name):
    imsize = np.array([imheight, imwidth])
    dims = {
        "coarse": imsize // 16,
        "medium": imsize // 8,
        "fine": imsize // 4,
    }
    feats_2d = {
        "coarse": torch.empty(
            (1, len(rgb_imgfiles), 80, *dims["coarse"]), dtype=torch.float16
        ),
        "medium": torch.empty(
            (1, len(rgb_imgfiles), 40, *dims["medium"]), dtype=torch.float16
        ),
        "fine": torch.empty(
            (1, len(rgb_imgfiles), 24, *dims["fine"]), dtype=torch.float16
        ),
    }
    cam_positions = torch.from_numpy(cam_positions).cuda()[None]
    for i in range(len(rgb_imgfiles)):
        rgb_img = data.load_rgb_imgs([rgb_imgfiles[i]], imheight, imwidth)
        rgb_img = torch.from_numpy(rgb_img).cuda()[None]
        cur_proj_mats = {k: v[:, i, None] for k, v in proj_mats.items()}
        if SRfeat:
            SRfeat0, SRfeat1, SRfeat2 = data.load_SRfeats(scene_name, ['0'+rgb_imgfiles[i][-9:-4]])
            SRfeat0 = torch.from_numpy(SRfeat0).cuda()[None]
            SRfeat1 = torch.from_numpy(SRfeat1).cuda()[None]
            SRfeat2 = torch.from_numpy(SRfeat2).cuda()[None]
            cur_feats_2d = model.cvrecon.get_SR_feats(SRfeat0, SRfeat1, SRfeat2, cur_proj_mats, cam_positions[:, i, None])
        else:
            cur_feats_2d = model.cvrecon.get_img_feats(rgb_img, cur_proj_mats, cam_positions[:, i, None])

        for resname in feats_2d:
            feats_2d[resname][0, i] = cur_feats_2d[resname][0, 0].cpu()
    return feats_2d


def construct_cv(model, cur_feats, ref_feats, intr, pose, ref_pose, n_imgs, rgb_imgfiles):
    cvs = []
    cv_masks = []

    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intr
    k[0] = k[0] * 0.125
    k[1] = k[1] * 0.125
    invK = torch.from_numpy(np.linalg.inv(k)).unsqueeze(0).cuda()
    k = torch.from_numpy(k).cuda()

    src_K = k.unsqueeze(0).unsqueeze(0).repeat(1, 7, 1, 1)  # [1, 7, 4, 4]
    min_depth = torch.tensor(0.25).type_as(src_K).view(1, 1, 1, 1)
    max_depth = torch.tensor(5.0).type_as(src_K).view(1, 1, 1, 1)

    inv_pose = torch.from_numpy(np.linalg.inv(pose)).cuda()
    inv_ref_pose = torch.from_numpy(np.linalg.inv(ref_pose)).cuda().view(n_imgs, 7, 4, 4)
    pose = torch.from_numpy(pose).cuda()
    ref_pose = torch.from_numpy(ref_pose).cuda().view(n_imgs, 7, 4, 4)

    if vis_lowest:
        output_path = f'/test/{scene_name}'
        if not os.path.exists(output_path): os.mkdir(output_path)
        gif = []

    for i in range(n_imgs):
        matching_cur_feats = cur_feats[i].unsqueeze(0)
        matching_src_feat = ref_feats[i].unsqueeze(0)

        src_cam_T_world = inv_ref_pose[i].unsqueeze(0)
        src_world_T_cam = ref_pose[i].unsqueeze(0)
        cur_cam_T_world = inv_pose[i].unsqueeze(0)
        cur_world_T_cam = pose[i].unsqueeze(0)
        with torch.cuda.amp.autocast(False):
            # Compute src_cam_T_cur_cam, a transformation for going from 3D 
            # coords in current view coordinate frame to source view coords 
            # coordinate frames.
            src_cam_T_cur_cam = src_cam_T_world @ cur_world_T_cam.unsqueeze(1)

            # Compute cur_cam_T_src_cam the opposite of src_cam_T_cur_cam. From 
            # source view to current view.
            cur_cam_T_src_cam = cur_cam_T_world.unsqueeze(1) @ src_world_T_cam
        
        cost_volume, lowest_cost, _, overall_mask_bhw = model.cvrecon.cost_volume(
                                            cur_feats=matching_cur_feats,
                                            src_feats=matching_src_feat,
                                            src_extrinsics=src_cam_T_cur_cam,
                                            src_poses=cur_cam_T_src_cam,
                                            src_Ks=src_K,
                                            cur_invK=invK,
                                            min_depth=min_depth,
                                            max_depth=max_depth,
                                            return_mask=True,
                                            return_lowest=vis_lowest,
                                        )

        if vis_lowest:
            lowest_cost_3hw = colormap_image(lowest_cost, vmin=5, vmax=0.25)
            gif.append(Image.fromarray(np.uint8(lowest_cost_3hw.permute(1,2,0).cpu().detach().numpy() * 255)))

        cvs.append(cost_volume.unsqueeze(1))
        cv_masks.append(overall_mask_bhw.unsqueeze(1))

    if vis_lowest:
        gif[0].save(os.path.join(output_path, 'lowest_cost.gif'), save_all=True,optimize=False, append_images=gif[1:], loop=0)
        save_gif(rgb_imgfiles, output_path)

    cvs = torch.cat(cvs, dim=1)  # [b, n, c, d, h, w]
    cv_masks = torch.cat(cv_masks, dim=1)
    if config["cv_overall"]:
        ############################### skiped overall feat ####################################################
        # overallfeat = cvs[:, :, -1:, ::8, ...].permute(0, 1, 3, 2, 4, 5).expand([-1, -1, -1, cvs.shape[3], -1, -1])

        # ############################### conv overall feat ####################################################
        # overallfeat = cvs[:, :, -1, ...].view([-1] + list(cvs.shape[3:]))
        # overallfeat = self.cv_global_encoder(overallfeat).view(list(cvs.shape[:2]) + [8, 1, cvs.shape[-2], cvs.shape[-1]])
        # overallfeat = overallfeat.expand([-1, -1, -1, cvs.shape[3], -1, -1])

        # ############################### complete overall feat ####################################################
        # overallfeat = cvs[:, :, -1:, :, ...].permute(0, 1, 3, 2, 4, 5).expand([-1, -1, -1, cvs.shape[3], -1, -1])

        # cvs = cvs[:, :, :-1, ...]
        # cvs = torch.cat([overallfeat, cvs], dim=2)
        pass
    return cvs, cv_masks


def inference(model, info_file, outfile, n_imgs, cropsize, SRlist=None, scene_name=None, CVDict=None):
    rgb_imgfiles, depth_imgfiles, pose, intr = load_scene(info_file)
    test_img = imageio.imread(rgb_imgfiles[0])
    imheight, imwidth, _ = test_img.shape

    scene_minbound, scene_maxbound = get_scene_bounds(
        pose, intr, imheight, imwidth, frustum_depth=4
    )

    pose_w2c = np.linalg.inv(pose)
    tiles = get_tiles(  # divide to non-overlapping fragments
        scene_minbound,
        scene_maxbound,
        cropsize_voxels_fine=np.array(cropsize),
        voxel_size_fine=0.04,
    )

    # pre-select views for each tile
    tiles = frame_selection(
        tiles, pose, intr, imheight, imwidth, n_imgs=n_imgs, rmin_deg=15, tmin=0.1, SRlist=SRlist, rgb_imgfiles=rgb_imgfiles, CVDict=CVDict,
    )

    # drop the frames that weren't selected for any tile, re-index the selected frame indicies
    selected_frame_inds = np.unique(
        np.concatenate([tile["frame_inds"] for tile in tiles])
    )

    all_frame_inds = np.arange(len(pose))
    frame_reindex = np.full(len(all_frame_inds), 100_000)
    frame_reindex[selected_frame_inds] = np.arange(len(selected_frame_inds))
    for tile in tiles:
        tile["frame_inds"] = frame_reindex[tile["frame_inds"]]
    pose_w2c = pose_w2c[selected_frame_inds]
    pose = pose[selected_frame_inds]
    rgb_imgfiles = np.array(rgb_imgfiles)[selected_frame_inds]

    if CVDict is not None:
        with open(info_file, "r") as f:
            info = json.load(f)
        ref_pose = []
        ref_img = []
        cv_invalid_mask = np.zeros(len(rgb_imgfiles), dtype=np.int)
        frame2id = {'0'+frame["filename_color"][-9:-4]:i for i, frame in enumerate(info["frames"])}
        for i, fname in enumerate(rgb_imgfiles.copy()):
            if '0' + fname[-9: -4] in CVDict:
                for frameid in CVDict['0' + fname[-9: -4]]:
                    ref_pose.append(np.array(info['frames'][frame2id[frameid]]['pose'], dtype=np.float32)[None,...])
                    ref_img.append(info['frames'][frame2id[frameid]]['filename_color'])
            else:
                print('!!!!!!!!!!!!!! invalid cv at ', '0' + fname[-9: -4])
                cv_invalid_mask[i] = 1
                for i in range(7):
                    ref_pose.append(np.array(info['frames'][frame2id['0'+fname[-9: -4]]]['pose'], dtype=np.float32)[None,...])
                    ref_img.append(fname)
        ref_pose = np.concatenate(ref_pose)
        ref_imgs_paths = np.array(ref_img)

        cur_imgs = []
        for i in range(len(rgb_imgfiles)):
            cur_img = data.load_rgb_imgs([rgb_imgfiles[i]], imheight, imwidth)
            cur_imgs.append(torch.from_numpy(cur_img).cuda()[None])
        cur_imgs = torch.cat(cur_imgs, dim=0)
        cur_feats = model.cvrecon.compute_matching_feats(cur_imgs).squeeze()  # [n_imgs, c, h, w]

        ref_imgs = []
        for i in range(len(ref_imgs_paths)):
            ref_img = data.load_rgb_imgs([ref_imgs_paths[i]], imheight, imwidth)
            ref_imgs.append(torch.from_numpy(ref_img).cuda())
        ref_imgs = torch.cat(ref_imgs, dim=0).view(-1, 7, 3, imheight, imwidth)  # [n_imgs, 7, 3, h, w]
        ref_feats = model.cvrecon.compute_matching_feats(ref_imgs)  # [n_imgs, 7, c, h, w]

        cost_volume, cv_masks = construct_cv(model, cur_feats, ref_feats, intr, pose, ref_pose, len(rgb_imgfiles), rgb_imgfiles)
        cost_volume[0][cv_invalid_mask.astype(bool)] = 0

    factors = np.array([1 / 16, 1 / 8, 1 / 4])
    proj_mats = data.get_proj_mats(intr, pose_w2c, factors)
    proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}
    img_feats = get_img_feats(
        model,
        imheight,
        imwidth,
        proj_mats,
        rgb_imgfiles,
        cam_positions=pose[:, :3, 3],
        SRfeat=SRlist!=None,
        scene_name=scene_name,
    )
    for resname, res in model.cvrecon.resolutions.items():

        # populate feature volume independently for each tile
        for tile in tiles:
            voxel_coords = tile["voxel_coords"].cuda()
            voxel_batch_inds = torch.zeros(
                len(voxel_coords), dtype=torch.int64, device="cuda"
            )

            cur_img_feats = img_feats[resname][:, tile["frame_inds"]].cuda()
            cur_proj_mats = proj_mats[resname][:, tile["frame_inds"]]
            cur_cost_volume = cost_volume[:, tile["frame_inds"]].clone()
            featheight, featwidth = img_feats[resname].shape[-2:]
            
            #################################################### 2dfeat & CV group conv ################################################################
            feat_cha = {'coarse': 80, 'medium': 40, 'fine':24}
            cv_dim = 15 - 8
            bs = 1
            if resname != 'medium':
                cur_cost_volume = F.interpolate(cur_cost_volume.view([bs*n_imgs*cv_dim, 64, 60, 80]), [featheight, featwidth]).view([bs, n_imgs, cv_dim, 64, featheight, featwidth])
            cur_img_feats = model.cvrecon.cv_global_encoder[resname](torch.cat([cur_cost_volume[:,:,-1], cur_img_feats], dim=2).view([-1, feat_cha[resname]+64, featheight, featwidth]))
            cur_img_feats = cur_img_feats.view([bs, n_imgs, feat_cha[resname]+64, featheight, featwidth])
            overallfeat = cur_img_feats[:, :, :64].unsqueeze(3).expand([-1, -1, -1, 64, -1, -1])
            cur_img_feats = cur_img_feats[:,:,64:]
            cur_cost_volume = model.cvrecon.unshared_conv[resname](
                torch.cat([cur_img_feats.unsqueeze(3).expand([-1,-1,-1,64,-1,-1]), cur_cost_volume], dim=2).transpose(2,3).reshape(bs*n_imgs,-1,featheight, featwidth))
            cur_cost_volume = cur_cost_volume.view([bs, n_imgs, 64, 7, featheight, featwidth]).transpose(2,3)
            cur_cost_volume = torch.cat([overallfeat, cur_cost_volume], dim=2)
            #############################################################################################################################################

            bp_uv, bp_depth, bp_mask = model.cvrecon.project_voxels(
                voxel_coords,
                voxel_batch_inds,
                cur_proj_mats.transpose(0, 1),
                featheight,
                featwidth,
            )
            bp_data = {
                "voxel_batch_inds": voxel_batch_inds,
                "bp_uv": bp_uv,
                "bp_depth": bp_depth,
                "bp_mask": bp_mask,
            }
            bp_feats, proj_occ_logits = model.cvrecon.back_project_features(
                bp_data,
                cur_img_feats.transpose(0, 1),
                model.cvrecon.mv_fusion[resname],
                cur_cost_volume if (CVDict is not None) else None,
                cv_masks[:, tile["frame_inds"]] if (CVDict is not None) else None,
            )
            bp_feats = model.cvrecon.layer_norms[resname](bp_feats)

            tile["voxel_features"] = torch.cat(
                (tile["voxel_features"], bp_feats.cpu(), tile["voxel_logits"]),
                dim=-1,
            )

        # combine all tiles into one sparse tensor & run convolution
        voxel_inds = torch.cat([tile["voxel_inds"] for tile in tiles], dim=0)
        voxel_batch_inds = torch.zeros((len(voxel_inds), 1), dtype=torch.int32)
        voxel_features = torchsparse.SparseTensor(
            torch.cat([tile["voxel_features"] for tile in tiles], dim=0).cuda(),
            torch.cat([voxel_inds, voxel_batch_inds], dim=-1).cuda(),
        )

        voxel_features = model.cvrecon.cnns3d[resname](voxel_features)
        voxel_logits = model.cvrecon.output_layers[resname](voxel_features)

        if resname in ["coarse", "medium"]:
            # sparsify & upsample
            occupancy = voxel_logits.F.squeeze(1) > 0
            if not torch.any(occupancy):
                raise Exception("um")
            voxel_features = model.cvrecon.upsampler.upsample_feats(
                voxel_features.F[occupancy]
            )
            voxel_inds = model.cvrecon.upsampler.upsample_inds(voxel_logits.C[occupancy])
            voxel_logits = model.cvrecon.upsampler.upsample_feats(
                voxel_logits.F[occupancy]
            )
            voxel_features = voxel_features.cpu()
            voxel_inds = voxel_inds.cpu()
            voxel_logits = voxel_logits.cpu()

            # split back up into tiles
            for tile in tiles:
                tile["origin_ind"] *= 2
                tile["maxbound_ind"] *= 2

                tile_voxel_mask = (
                    (voxel_inds[:, 0] >= tile["origin_ind"][0])
                    & (voxel_inds[:, 1] >= tile["origin_ind"][1])
                    & (voxel_inds[:, 2] >= tile["origin_ind"][2])
                    & (voxel_inds[:, 0] < tile["maxbound_ind"][0])
                    & (voxel_inds[:, 1] < tile["maxbound_ind"][1])
                    & (voxel_inds[:, 2] < tile["maxbound_ind"][2])
                )

                tile["voxel_inds"] = voxel_inds[tile_voxel_mask, :3]
                tile["voxel_features"] = voxel_features[tile_voxel_mask]
                tile["voxel_logits"] = voxel_logits[tile_voxel_mask]
                tile["voxel_coords"] = tile["voxel_inds"] * (
                    res / 2
                ) + scene_minbound.astype(np.float32)

    tsdf_vol = utils.to_vol(
        voxel_logits.C[:, :3].cpu().numpy(),
        1.05 * torch.tanh(voxel_logits.F).squeeze(-1).cpu().numpy(),
    )
    mesh = utils.to_mesh(
        -tsdf_vol,
        voxel_size=0.04,
        origin=scene_minbound,
        level=0,
        mask=~np.isnan(tsdf_vol),
    )
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default='test', type=str)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--use-proj-occ", default=True, type=bool)
    parser.add_argument("--n-imgs", default=60, type=int)
    parser.add_argument("--cropsize", default=96, type=int)
    parser.add_argument('--vis-lowest', action='store_true')
    args = parser.parse_args()

    pl.seed_everything(0)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cropsize = (args.cropsize, args.cropsize, 48)

    SRfeat = config["SRfeat"]
    useCV = config["cost_volume"]
    vis_lowest = True if args.vis_lowest else False
    if SRfeat:
        from collections import defaultdict
        SRlists = defaultdict(list)
        with open('/data_splits/ScanNetv2/standard_split/{}_eight_view_deepvmvs_dense_for_cvrecon.txt'.format(args.split), 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            scan_id, frame_id = line.split(" ")[:2]
            SRlists[scan_id].append(frame_id)
    
    if useCV:
        from collections import defaultdict
        CVDicts = defaultdict(dict)
        fname = '/data_splits/ScanNetv2/standard_split/{}_for_cvrecon.txt'.format(args.split)
        if args.split == 'test':
            fname = '/data_splits/ScanNetv2/standard_split/test_eight_view_deepvmvs_dense_for_cvrecon.txt'
        with open(fname, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            scan_id, *frame_id = line.split(" ")
            CVDicts[scan_id][frame_id[0]] = frame_id[1:]

    with torch.cuda.amp.autocast():

        info_files = utils.load_info_files(config["scannet_dir"], args.split)
        model = load_model(args.ckpt, args.use_proj_occ, config)
        for info_file in tqdm.tqdm(info_files):

            scene_name = os.path.basename(os.path.dirname(info_file))
            outdir = os.path.join(args.outputdir, scene_name)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, "prediction.ply")

            # if os.path.exists(outfile):
            #     print(outfile, 'exists, skipping')
            #     continue

            # try:
            if SRfeat:
                mesh = inference(model, info_file, outfile, args.n_imgs, cropsize, SRlists[scene_name], scene_name)
            elif useCV:
                mesh = inference(model, info_file, outfile, args.n_imgs, cropsize, CVDict=CVDicts[scene_name])
            else:
                mesh = inference(model, info_file, outfile, args.n_imgs, cropsize)
            o3d.io.write_triangle_mesh(outfile, mesh)
            # except Exception as e:
            #     print(e)
