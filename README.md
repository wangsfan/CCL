## Introduction
This is our implementation of our paper *Causal Contrastive Learning for Unsupervised Graph Domain Adaptation*.

**TL;DR**: CCL disentangles graph embeddings into causal and spurious components via mutual information constraints and cross-domain feature intervention, enabling robust knowledge transfer from labeled source graphs to unlabeled target graphs.

**Abstract**: Unsupervised graph domain adaptation transfers knowledge from labeled source graphs to unlabeled target graphs under distribution shifts. Due to the complicated underlying causality of cross-domain adaptation on graphs, Existing methods suffer from alignment collapse and poor generalization among different domains. By using causal analysis and contrastive learning as two powerful tools, we design a generative causal intervention mechanism to disentangle domain-invariant causal feature and domain-specific spurious feature. Typically, we propose a Causal Contrastive Learning framework (CCL) with three modules: Contrastive Alignment module, Feature Decomposition module and Causal Decoupling module. As the core component, the Causal Decoupling module decouples causal feature from three aspects: (a) Causal feature disentanglement by modeling the mutual information between causal representation and labels; (b) Spurious feature disentanglement by modeling the mutual information between spurious representation, spurious structure and labels; (c) Adversarial reconstruction by generating confounding graph embedding incurred from interchanged source data and target data. In essence, CCL accomplishes stable knowledge transfer from labeled source  graphs to unlabeled target graphs by suppressing spurious correlations via mutual information modeling and cross-domain feature intervention. Extensive experiments on eight UGDA benchmarks demonstrate that CCL significantly outperforms state-of-the-art approaches for unsupervised graph domain adaptation. 


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

