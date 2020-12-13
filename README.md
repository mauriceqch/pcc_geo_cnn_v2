# Improved Deep Point Cloud Geometry Compression

<p align="center">
  <img src="image.png?raw=true" alt="Comparison samples"/>
</p>


* **Authors**:
[Maurice Quach](https://scholar.google.com/citations?user=atvnc2MAAAAJ),
[Giuseppe Valenzise](https://scholar.google.com/citations?user=7ftDv4gAAAAJ) and
[Frederic Dufaux](https://scholar.google.com/citations?user=ziqjbTIAAAAJ)  
* **Affiliation**: Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, 91190 Gif-sur-Yvette, France
* **Funding**: ANR ReVeRy national fund (REVERY ANR-17-CE23-0020)
* **Links**: [[Paper]](https://arxiv.org/abs/2006.09043)

## Experimental data

We provide the full experimental data used for the paper. This includes:
- In `models`, all trained models
- In `results/data.csv`, bitrates and objective metric values for all models, point clouds and metrics (D1/D2)
- Compressed and decompressed point clouds for all models (c1 to c6, G-PCC trisoup, G-PCC octree)

[Download experimental data](https://drive.google.com/file/d/18uHmr0ZpgFLeL9Y5TUFTsQkRfz4XpQdJ/view?usp=sharing)

## Prerequisites

* Python 3.6.9
* Tensorflow 1.15.0 with CUDA 10.0.130 and cuDNN 7.4.2
* [tensorflow-compression](https://github.com/tensorflow/compression) 1.3
* MPEG G-PCC codec [mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13): necessary only to compare results with G-PCC,
to obtain more recent versions you need to register on the [MPEG Gitlab](http://mpegx.int-evry.fr/software/MPEG/PCC/) and request the permissions for `MPEG/PCC`
* MPEG metric software v0.12.3 [mpeg-pcc-dmetric](http://mpegx.int-evry.fr/software/MPEG/PCC/mpeg-pcc-dmetric):
available on the [MPEG Gitlab](http://mpegx.int-evry.fr/software/MPEG/PCC/), you need to register and request the permissions for `MPEG/PCC`
* MPEG PCC dataset: refer to Common Test Conditions (CTCs) to download the full dataset,
you can also get some point clouds from [JPEG Pleno](http://plenodb.jpeg.org/).
* packages in `requirements.txt`

*Note 1*: using a Linux distribution such as Ubuntu is highly recommended  
*Note 2*: CTCs can be found at [wg11.sc29.org](http://wg11.sc29.org) in
All Meetings > Latest Meeting > Output documents "Common test conditions for point cloud compression".
For example, "Common test conditions for PCC", in ISO/IEC JTC1/SC29/WG11 MPEG output document N19324 is in the Alpbach meeting 130.

## Getting started

### Configuration

Adapt the configurations in `ev_experiment.yml` to your particular setup:
* `MPEG_TMC13_DIR`: G-PCC folder (`mpeg-pcc-tmc13`)
* `PCERROR`: `mpeg-pcc-dmetric` folder
* `MPEG_DATASET_DIR`: MPEG PCC dataset folder
* `TRAIN_DATASET_PATH`: Path to training dataset
* `TRAIN_RESOLUTION`: Resolution of the training dataset
* `EXPERIMENT_DIR`: Experiment folder, all results are saved in this folder, it needs to be created manually

### Datasets

First, download the ModelNet40 manually aligned dataset: [http://modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu).  
Then, we generate the training dataset specified in our paper (block size 64) with the following commands:

    python ds_select_largest.py ~/data/datasets/ModelNet40 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200 200
    python ds_mesh_to_pc.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512 --vg_size 512
    python ds_pc_octree_blocks.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3 --vg_size 512 --level 3 
    python ds_select_largest.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3_4k 4000

### Run experiments

Run G-PCC experiments:

    python mp_run.py ev_experiment.yml

*Note 1*: It is **highly** recommended to set `EXPERIMENT_DIR` on an SSD drive.
On an HDD, it is recommended to add the `--num_parallel 1` option as it can be extremely slow otherwise.  
    
Train all models, run experiments and compare results:

    python tr_train_all.py ev_experiment.yml && \
    python ev_run_experiment.py ev_experiment.yml --num_parallel 8 && \
    python ev_run_compare.py ev_experiment.yml

*Note 1*: In the provided configuration `ev_experiment.yml`, a **large** number of models are trained. This takes about 4 days on an Nvidia GeForce GTX 1080 Ti.
*Note 2*: Adjust the `--num_parallel` option for `ev_run_experiment.py` depending on your GPU memory.
With 11GB, we've found 8 to leave a fair amount of GPU memory available.

To build the LaTeX tables and gather the figures:

    python ut_build_paper.py ev_experiment.yml figs/

To render the point clouds and the visual comparisons:

    python ut_run_render.py ev_experiment.yml

To draw the training plots from tfevents data:

    python ut_tensorboard_plots.py ev_experiment.yml

## Overview

    ├── requirements.txt                            package requirements
    └── src
        ├── compress_octree.py                      [Coding] Compress a point cloud
        ├── decompress_octree.py                    [Coding] Decompress a point cloud
        ├── ds_mesh_to_pc.py                        [Dataset] Convert mesh to point cloud
        ├── ds_pc_octree_blocks.py                  [Dataset] Divide a point cloud into octree blocks
        ├── ds_select_largest.py                    [Dataset] Select the N largest files from a folder
        ├── ev_compare.py                           [Eval] Compare results (curves, BD-Rates and BD-PSNRs...)
        ├── ev_experiment.py                        [Eval] Run pipeline for a point cloud (compress, decompress, eval)
        ├── ev_experiment.yml                       [Config] Experimental configuration
        ├── ev_run_compare.py                       [Eval] Compare results for the experimental configuration
        ├── ev_run_experiment.py                    [Eval] Run pipelines for the experimental configuration
        ├── map_color.py                            [Utils] Transfer colors from a point cloud onto another
        ├── model_configs.py                        [Model] Model configurations
        ├── model_opt.py                            [Model] Threshold optimization
        ├── model_syntax.py                         [Model] Bitsream specification
        ├── model_transforms.py                     [Model] Transforms configurations
        ├── model_types.py                          [Model] Model types
        ├── mp_report.py                            [MPEG] Generate JSON report for a G-PCC folder
        ├── mp_run.py                               [MPEG] Run G-PCC for the experimental configuration
        ├── tr_train.py                             [Train] Train a compression model
        ├── tr_train_all.py                         [Train] Train compression models for the experimental configuration
        ├── ut_build_paper.py                       [Utils] Create LaTeX tables and gather figures
        ├── ut_run_render.py                        [Utils] Render point clouds and visual comparisons
        ├── ut_tensorboard_plots.py                 [Utils] Generate plots from tfevents data
        └── utils
            ├── bd.py                               BD-Rate and BD-PSNR computation
            ├── colorbar.py                         Colorbar generation
            ├── experiment.py                       Experimental utilities
            ├── focal_loss.py                       Focal loss
            ├── matplotlib_utils.py                 Matplotlib utilies
            ├── mpeg_parsing.py                     MPEG log files parsing
            ├── o3d.py                              Open3D utilities (rendering)
            ├── octree_coding.py                    Octree coding
            ├── parallel_process.py                 Parallel processing
            ├── patch_gaussian_conditional.py       Patch tensorflow-compression GaussianConditional to get debug tensors
            ├── pc_io.py                            Point Cloud Input/Output
            ├── pc_metric.py                        Point Cloud geometry distortion metrics (D1 and D2)
            ├── pc_to_camera_params.py              Generate camera parameters for a Point Cloud
            └── pc_to_img.py                        Convert a Point Cloud to an image using predefined camera parameters

## Citation

    @misc{quach2020improved,
        title={Improved Deep Point Cloud Geometry Compression},
        author={Maurice Quach and Giuseppe Valenzise and Frederic Dufaux},
        year={2020},
        eprint={2006.09043},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

