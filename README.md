# federated_learning

In this project, we address key challenges in federated learning for medical image analysis, focusing on communication costs, data heterogeneity, and reliance on supervised learning. Federated learning enables collaborative model training across institutions without sharing raw data, but the size and complexity of models in medical image segmentation often lead to high communication costs during model aggregation. Additionally, the non-IID nature of medical data, caused by differences in equipment, demographics, and protocols, hampers global model convergence. The reliance on labeled data, which is costly and time-intensive to annotate, further limits its scalability.

Using the spleen dataset from the Medical Segmentation Decathlon (MSD), consisting of 61 3D CT scans, we implemented horizontal federated learning. The dataset was divided to simulate multiple healthcare institutions, introducing both IID and non-IID conditions by varying data distributions. For the segmentation task, we employed the UNet architecture using the MONAI framework.

We began with a baseline FedAvg algorithm and compared its performance to a centralized model. To address data heterogeneity, we implemented FedCluster, which aggregates models considering client variability. To reduce communication costs, we integrated knowledge distillation, which minimizes communication overhead. Finally, we explored FedMix, a mixed-supervised framework leveraging both labeled and pseudo-labeled data, to enhance segmentation performance under federated settings.

The dataset is available here: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

## Project structure
---
#### **`data/raw/`**
Once the data has been downloaded, this directory should contain the raw dataset files.

---

#### **`models_gpu/`** and **`models_i5/`**
These directories store the models trained on a  a system equipped with an NVIDIA RTX4060 GPU, and a system with a 2 GHz Quad-Core Intel Core i5 processor, respectively. 

---

#### **`notebooks/`**
This directory contains the Jupyter Notebooks used for implementing and analyzing various federated learning approaches:
- **`centralized_learning.ipynb`**: Implements a centralized training pipeline to compare with federated learning models.
- **`fedavg_KD.ipynb`**: Implements the FedAvg algorithm with knowledge distillation to reduce communication costs.
- **`fedcluster_KD.ipynb`**: Implements FedCluster, which considers data heterogeneity, along with knowledge distillation.
- **`federated_learning.ipynb`**: The base implementation of the federated learning framework with FedAvg and FedCluster.
- **`fedmixcluster.ipynb`**: Implements the FedMix algorithm for mixed-supervised federated learning, considering both labeled and pseudo-labeled data.
- **`plot_fig2_fig3.ipynb`**: Scripts for generating plots related to Figure 2 and Figure 3 in the results.
- **`plot.ipynb`**: General plotting scripts for the rest of the figures in the report.

---

#### **`results/`**
This directory contains results generated during experimentation, including evaluation metrics and data for plotting:

---

### Summary
This repository provides a modular structure for implementing, testing, and analyzing federated learning approaches in medical image segmentation. Each notebook is self-contained, focusing on a specific technique or task, while the datasets, models, and results are organized for easy access and reproducibility.
