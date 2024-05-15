# Utilizing Vision Transformer (ViT) Models for End-to-End Vision-Based Quadrotor Obstacle Avoidance

Under construction; code for training and testing coming very soon, with instructions, pre-trained weights, and datasets.

This is the official repository for the paper "Utilizing vision transformer models for end-to-end vision-based quadrotor obstacle avoidance" by Bhattacharya, et al. (2024) from GRASP, Penn.

<font size="3"><u>[Project page](https://www.anishbhattacharya.com/research/vit-depthfly)</u></font>

We demonstrate that vision transformers (ViTs) can be used for end-to-end perception-based obstacle avoidance for quadrotors equipped with a depth camera. We train policies that predict linear velocity commands to avoid static obstacles via behavior cloning from a privileged expert in a simple simulation environment, and show that ViT models combined with recurrence layers (LSTMs) outperform baseline methods based on other popular architectures. Deployed on a real quadrotor, we can achieve zero-shot dodging behavior at speeds reaching 7 m/s.

<!-- GIFs -->

#### Generalization to simulation environments

<!-- insert gif here form media folder -->
<!-- ![Generalization to simulation trees](media/vitlstm_trees.gif) -->
<img src="media/trees-vitlstm.gif" width="400" height="200">

<!-- ![Generalization to window gap](media/walls-vitlstm.gif) -->
<img src="media/walls-vitlstm.gif" width="250" height="200">
<br>

#### Zero-shot transfer to multi-obstacle and high-speed, real-world dodging

<img src="media/multi-obstacle-vitlstm.gif" width="300" height="200">

<img src="media/7ms-vitlstm.gif" width="380" height="200">


## Installation

Coming really soon!

## Test

#### Pretrained weights

Visit the datashare link below and download `pretrained_models.tar`. This tarball includes pretrained models for ConvNet, LSTMnet, UNet, ViT, and ViT+LSTM (our best model).

[Datashare](https://upenn.app.box.com/v/ViT-quad-datashare)

## Train

Coming pretty soon!

## Citation

Coming quite soon!

## Acknowledgements

Simulation launching code and the versions of `flightmare` and `dodgedrone_simulation` are from the [ICRA 2022 DodgeDrone Competition code](https://github.com/uzh-rpg/agile_flight).
