## Introduction
This is our implementation of our paper *Causal Contrastive Learning for Unsupervised Graph Domain Adaptation*.

**TL;DR**: CCL disentangles graph embeddings into causal and spurious components via mutual information constraints and cross-domain feature intervention, enabling robust knowledge transfer from labeled source graphs to unlabeled target graphs.

**Abstract**: Unsupervised graph domain adaptation (UGDA) transfers knowledge from labeled source graphs to unlabeled target graphs under distribution shifts. Existing methods typically rely on global domain alignment while ignoring the entanglement of causal and spurious features, leading to alignment collapse and poor cross-domain generalization. Moreover, most approaches only use single-view graph encoders, which fail to capture comprehensive topological semantics for robust adaptation. To address these issues, we propose the Causal Contrastive Learning framework (CCL) with two ingredient modules: Contrastive Alignment module and Causal Decoupling module. The Contrastive Alignment module fuses complementary structural information from dual graph encoders to facilitate the robust representation of target graphs. The Causal Decoupling module explicitly disentangles domain-invariant causal features and domain-specific spurious features by mutual information modeling and interchanged adversary, which effectively avoid spurious correlation interference. 


## Dependencies
- torch>=1.13
- torch-geometric>=2.3
- numpy=1.24.3
- scikit-learn=1.6.1
- tqdm=4.67.1

## Usage

##### 1. Install dependencies
First we recommend to create a conda environment with all the required packages by using the following command.
```
conda create -n ccl python=3.9
```
This command creates a conda environment named `ccl`. You can activate the conda environment with the following command:
```
conda activate ccl
```
In the following sections, we assume that you use this conda environment or you manually install the required packages.
If you see the following error, you may need to install a PyTorch Geometric package compatible with your PyTorch version.
```
RuntimeError: No CUDA GPUs are available or ImportError: libcusparse.so.11: cannot open shared object file: No such file or directory
```
For example, if your infrastructure only supports CUDA == 11.3, you may need to install PyTorch and PyTorch Geometric with the corresponding CUDA version.
```
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric
```

##### 2. Run code
- Mutagenicity (12 domain pairs constructed by edge density)
    ```
    python mutagenicity/train.py
    ```

- PTC (12 domain pairs from four PTC subsets)
    ```
    python ptc/train.py
    ```

- COX2 (cross-dataset: COX2_MD â†?COX2)
    ```
    python cox2/train.py
    ```

- PROTEINS â†?DD (cross-dataset with different feature dimensions)
    ```
    python proteins_dd/train.py
    ```

- BZR_MD â†?BZR (cross-dataset with different feature dimensions)
    ```
    python bzr/train.py
    ```


## Parameters

| Parameter         |           Description                       |
|-------------------|---------------------------------------------|
| hidden_dim        |   Hidden dimension of GIN layers            |
| output_dim        |   Dimension of graph embeddings z1, z2      |
| num_layers        |   Number of GIN layers                      |
| z_c_dim           |   Dimension of causal embedding c           |
| z_s_dim           |   Dimension of spurious embedding c'        |
| batch_size        |   Batch size for training                   |
| lr                |   Learning rate                             |
| epochs            |   Number of training epochs                 |
| warmup_epochs     |   Warm-up epochs (source-only supervision)  |
| alpha             |   Loss weight for (L_CDC + L_MI^1 + L_MI^2) |
| beta              |   Loss weight for (L_inter)                 |
| source_train_ratio|   Train ratio of source domain              |
| target_train_ratio|   Train ratio of target domain              |
| seed              |   Random seed                               |
| device            |   GPU device (auto-detected)                |

