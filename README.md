# TCGA-LUAD Tumor Severity Prediction Using RNA-Seq and AI Models (ML & CNN)

This project predicts **tumor severity** in lung adenocarcinoma patients using gene expression data from the TCGA-LUAD dataset (retrieved from cBioPortal). We classify patients as:

- **Severe**: pathological stage > 1  
- **Non-severe**: pathological stage = 1

The dataset contains 510 patients, with approximately **55% severe** and **45% non-severe** cases.

> **Best accuracy achieved: 79%** (CNN model with 25% DEGs)  
> **Fully containerized with Docker**  
> **Nextflow pipeline** included for future HPC-based hyperparameter optimization

---

##Project Structure

```
TCGA-MLDL/
├── data/                         # Raw input data
│   ├── data_clinical_patient.txt
│   └── data_mrna_seq_tpm.txt
│
├── nf/                           # Nextflow pipeline (for future hyperparameter optimization)
│   ├── cnn_optuna_eval.py
│   ├── main.nf
│   ├── main.sh
│   └── nextflow.config
│
├── notebooks/                    # Analysis notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_splitting.ipynb
│   ├── 05_logistic_regr.ipynb
│   ├── 06_random_forest.ipynb
│   ├── 07_XGboost.ipynb          # ignore for this project
│   ├── 08_svm.ipynb
│   ├── 09_CNN.ipynb
│   ├── 10_results.ipnb
│
├── results/                      # Plots and outputs
├── clean_env.yml                 # Optional conda environment
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Models and Results

We evaluated 4 models across multiple splits (top 5%, 15%, 25% DEGs):

| Split     | Logistic | SVM   | RF    | CNN   |
|-----------|----------|-------|-------|--------|
| 5% DEGs   | 0.649    | 0.674 | 0.659 | 0.700 |
| 15% DEGs  | 0.767    | 0.703 | 0.662 | 0.730 |
| 25% DEGs  | 0.790    | 0.703 | 0.637 | **0.790** |

Final comparison: see `results/model_accuracies.png`

---

## How to Run with Docker

### 1. Build the Docker image

```bash
docker build -t tumor-severity-ml .
```

### 2. Launch Jupyter Notebook

```bash
docker run -p 8888:8888 -v $(pwd):/app tumor-severity-ml
```

Open the link from the terminal (localhost:8888).

---

## Nextflow Pipeline (future-ready)

A complete HPC pipeline for hyperparameter tuning (e.g., CNN + Optuna) is included in `/nf/`.

> Not yet used in this version but ready for scalable experimentation on SLURM or other clusters.

---

## Highlights

- Full ML pipeline from preprocessing to visualization
- Hyperparameter tuning with GridSearchCV and Optuna
- Final evaluation with grouped bar plot
- Dockerized for reproducibility
- Future SHAP-based interpretability planned

---

