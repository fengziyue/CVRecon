import torch

from cvrecon import transformer


class MVFusionMean(torch.nn.Module):
    def forward(self, features, valid_mask):
        return mv_fusion_mean(features, valid_mask)


class MVFusionTransformer(torch.nn.Module):
    def __init__(self, input_depth, n_layers, n_attn_heads, cv_cha=0):
        super().__init__()
        self.transformer = transformer.Transformer(
            input_depth,
            input_depth * 2,
            num_layers=n_layers,
            num_heads=n_attn_heads,
        )
        self.depth_mlp = torch.nn.Linear( 1 + cv_cha + 56, input_depth, bias=True)
        self.proj_tsdf_mlp = torch.nn.Linear(input_depth, 1, bias=True)

        for mlp in [self.depth_mlp, self.proj_tsdf_mlp]:
            torch.nn.init.kaiming_normal_(mlp.weight)
            torch.nn.init.zeros_(mlp.bias)

    def forward(self, features, bp_depth, bp_mask, use_proj_occ):
        '''
        features: [n_imgs, in_channels, n_voxels]
        bp_depth: [n_imgs, n_voxels]
        bp_mask: [n_imgs, n_voxels]
        '''
        device = features.device

        # attn_mask is False where attention is allowed.
        # set diagonal elements False to avoid nan
        attn_mask = bp_mask.transpose(0, 1)  # attn_mask [n_voxels, n_imgs]
        attn_mask = ~attn_mask[:, None].repeat(1, attn_mask.shape[1], 1).contiguous()  # [n_voxels, n_imgs, n_imgs]
        torch.diagonal(attn_mask, dim1=1, dim2=2)[:] = False

        im_z_norm = (bp_depth - 1.85) / 0.85
        features = torch.cat((features, im_z_norm[:, None]), dim=1)  # [n_imgs, in_channels+1, n_voxels]
        features = self.depth_mlp(features.transpose(1, 2))  # [n_imgs, n_voxels, in_channels]

        features = self.transformer(features, attn_mask)  # after self-attention still [n_imgs, n_voxels, in_channels]

        batchsize, nvoxels, _ = features.shape
        proj_occ_logits = self.proj_tsdf_mlp(
            features.reshape(batchsize * nvoxels, -1)
        ).reshape(batchsize, nvoxels)  # [n_imgs, n_voxels]

        if use_proj_occ:
            weights = proj_occ_logits.masked_fill(~bp_mask, -9e3)
            weights = torch.cat(
                (
                    weights,
                    torch.zeros(
                        (1, weights.shape[1]),
                        device=device,
                        dtype=weights.dtype,
                    ),
                ),
                dim=0,
            )
            features = torch.cat(
                (
                    features,
                    torch.zeros(
                        (1, features.shape[1], features.shape[2]),
                        device=device,
                        dtype=features.dtype,
                    ),
                ),
                dim=0,
            )
            weights = torch.softmax(weights, dim=0)
            pooled_features = torch.sum(features * weights[..., None], dim=0)
        else:
            pooled_features = mv_fusion_mean(features, bp_mask)

        return pooled_features, proj_occ_logits


def mv_fusion_mean(features, valid_mask):
    '''
    features: [n_imgs, n_voxels, n_channels]
    valid_mask: [n_imgs, n_voxels]

    return:
        pooled_features: each voxel's feature is the average of all seen pixels' feature
    '''
    weights = torch.sum(valid_mask, dim=0)
    weights[weights == 0] = 1
    pooled_features = (
        torch.sum(features * valid_mask[..., None], dim=0) / weights[:, None]
    )
    return pooled_features
