# Improve Motion Diffusion Model
## Quick Start
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13k2W21wlAKvPmMw6yp2ZCzuwtMmdu1ca?usp=sharing)

## Quick comparison to the base method MDM
### Quantitative evaluation
<div align="center">

<p align="center"><i>Table 1: Quantitative results on the <u>KIT</u> test set</i> (diffusion steps = 50)</p>

| **Model**    |                  **R Precision ⬆️**                          ||        |          **Diversity ➡️** |          **FID ⬇️**       |          **Download**       |
| ------------ | :-----------:        | :-----------:        | :-----------:         | :-----------:          | :-----------:          | :-----------:          |
|              |      Top 1           |     Top 2            |      Top 3            |                        |                        |                        |
|MDM           | $$0.3420^{\pm0066}$$ | $$0.5506^{\pm0055}$$ | $$0.6806^{\pm0038}$$  |  $$10.8756^{\pm0950}$$  |  $$1.5703^{\pm0658}$$  |  [🚀 Checkpoint](https://drive.google.com/file/d/1SHCRcE0es31vkJMLGf9dyLe7YsWj7pNL/view)  |
|*Ours          | $$\textcolor{yellow}{0.3465^{\pm0059}}$$ | $$\textcolor{yellow}{0.5568^{\pm0072}}$$ | $$\textcolor{yellow}{0.6910^{\pm0067}}$$  |  $$\textcolor{yellow}{11.3812^{\pm1153}}$$  |  $$\textcolor{yellow}{0.9819^{\pm0406}}$$  |  [🚀 Checkpoint](https://drive.google.com/file/d/1c-jXFakje0wtHpzWFUQerOKPCqHheBuU/view?usp=sharing)  |
</div>

<div align="center">

<p align="center"><i>Table 2: Quantitative results on the <u>HumanML3D</u> test set</i> (diffusion steps = 1000)</p>

| **Model**    |                  **R Precision ⬆️**                          ||        |          **Diversity ➡️** |          **FID ⬇️**       |          **Download**       |
| ------------ | :-----------:        | :-----------:        | :-----------:         | :-----------:          | :-----------:          | :-----------:          |
|              |      Top 1           |     Top 2            |      Top 3            |                        |                        |                        |
|MDM           | $$0.4141^{\pm.0091}$$ | $$0.6112^{\pm.0121}$$ | $$0.7248^{\pm.0076}$$  |  $$\textcolor{yellow}{9.7792^{\pm.1780}}$$  |  $$0.5970^{\pm.0984}$$  |  [🚀 Checkpoint](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view)  |
|*Ours          | $$\textcolor{yellow}{0.4304^{\pm.0128}}$$ | $$\textcolor{yellow}{0.6221^{\pm.0192}}$$ | $$\textcolor{yellow}{0.7326^{\pm.0127}}$$  |  $$9.6587^{\pm.2501}$$  |  $$\textcolor{yellow}{0.4563^{\pm.0511}}$$  |  [🚀 Checkpoint](https://drive.google.com/file/d/1rw-lbY6uVEkszaTXTg4KXRUJEMVplAkE/view?usp=sharing)  |
</div>



### Qualitative evaluation

<b>KIT</b>
<table class="center">
    <tr>
    <td>MDM</td>
    <td><img src="assets/mdm_1.gif"></td>
    <td><img src="assets/mdm_2.gif"></td>
    <td><img src="assets/mdm_3.gif"></td>
    <td><img src="assets/mdm_4.gif"></td>
    <td><img src="assets/mdm_5.gif"></td>
    </tr>
    <tr>
    <td>Ours</td>
    <td><img src="assets/ours_1.gif"></td>
    <td><img src="assets/ours_2.gif"></td>
    <td><img src="assets/ours_3.gif"></td>
    <td><img src="assets/ours_4.gif"></td>
    <td><img src="assets/ours_5.gif"></td>
    </tr>
</table>

<b>HUMANML3D</b>
<table class="center">
    <tr>
    <td>Text</td>
    <td>a person briskly walks around and falls then get up</td>
    <td>a person is crawling on all fours, and then gets up</td>
    <td>a person is doing a handstand</td>
    <td>a person walks forward, get pushed by someone, and he stumbles back</td>
    <td>losing balance, moving backwards with both feet</td>
    </tr>
    <tr>
    <td>MDM</td>
    <td><img src="assets/b_mdm_1.gif"></td>
    <td><img src="assets/b_mdm_2.gif"></td>
    <td><img src="assets/b_mdm_3.gif"></td>
    <td><img src="assets/b_mdm_4.gif"></td>
    <td><img src="assets/b_mdm_5.gif"></td>
    </tr>
    <tr>
    <td>Ours</td>
    <td><img src="assets/b_ours_1.gif"></td>
    <td><img src="assets/b_ours_2.gif"></td>
    <td><img src="assets/b_ours_3.gif"></td>
    <td><img src="assets/b_ours_4.gif"></td>
    <td><img src="assets/b_ours_5.gif"></td>
    </tr>
</table>



This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment 

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```

Setup conda env:
```shell
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 2. Get data
Dataset structure

      ├── dataset
        ├── HumanML3D
        │   ├── new_joint_vecs
        │   │   └── ...
        │   ├── new_joints
        │   │   └── ...
        │   ├── texts
        │   │   └── ...
        │   ├── Mean.npy
        │   ├── Std.npy
        │   ├── b_mdm_1.txt
        │   ├── train_val.txt
        │   ├── train.txt
        │   └── val.txt
        └── KIT-ML
            ├── new_joint_vecs
            │   └── ...
            ├── new_joints
            │   └── ...
            ├── texts
            │   └── ...
            ├── Mean.npy
            ├── Std.npy
            ├── b_mdm_1.txt
            ├── train_val.txt
            ├── train.txt
            └── val.txt
KIT
```bash
bash prepare/download_kit_dataset.sh
```
HumanML3D
```bash
bash prepare/download_humanml3d_dataset.sh
```
</details>

### 3. Train model
<details>
  <summary><b>Train your own MDM</b></summary>

```shell
python -m train.train_mdm --arch trans_enc --save_dir save/mdm_output --dataset kit
```
</details>

<details>
  <summary><b>Train your own LGTT</b></summary>
  
```shell
python -m train.train_mdm --arch lgtt --save_dir save/lgtt_output --dataset kit
```
</details>

<details>
  <summary><b>Train your own SinMDM</b></summary>
  
```shell
python -m train.train_mdm --arch unet --save_dir save/my_kit_unet_512 --dataset kit
```
</details>


## Motion Synthesis
```shell
python -m sample.generate --model_path ./save/kit_trans_enc_512/model000400000.pt --
```
## Evaluate

```shell
python -m eval.eval_humanml --model_path ./save/kit_trans_enc_512/model000400000.pt
```

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model), [guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).
