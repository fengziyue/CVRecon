# CVRecon: Rethinking 3D Geometric Feature Learning for Neural Reconstruction


### Dependencies

```
conda create -n cvrecon python=3.9 -y
conda activate cvrecon

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

pip install \
  pytorch-lightning==1.5 \
  scikit-image==0.18 \
  numba \
  pillow \
  wandb \
  tqdm \
  open3d \
  pyrender \
  ray \
  trimesh \
  pyyaml \
  matplotlib \
  black \
  pycuda \
  opencv-python \
  imageio

sudo apt install libsparsehash-dev
pip install torchsparse-v1.4.0 

pip install -e .
```


### Data

The ScanNet data should be downloaded and extracted by the script provided by the authors.


To format ScanNet for cvrecon:
```
python tools/preprocess_scannet.py --src path/to/scannet_src --dst path/to/new/scannet_dst
```
In `config.yml`, set `scannet_dir` to the value of `--dst`.

To generate ground truth tsdf:
```
python tools/generate_gt.py --data_path path/to/scannet_src --save_name TSDF_OUTPUT_DIR
# For the test split
python tools/generate_gt.py --test --data_path path/to/scannet_src --save_name TSDF_OUTPUT_DIR
```
In `config.yml`, set `tsdf_dir` to the value of `TSDF_OUTPUT_DIR`.

## Training

```
python scripts/train.py --config config.yml
```
Parameters can be adjusted in `config.yml`.
Set `attn_heads=0` to use direct averaging instead of transformers.

## Inference

```
python scripts/inference.py \
  --ckpt path/to/checkpoint.ckpt \
  --split [train / val / test] \
  --outputdir path/to/desired_output_directory \
  --n-imgs 60 \
  --config config.yml \
  --cropsize 96
```

## Evaluation

Refer to the evaluation protocal by Atlas and TransformerFusion
