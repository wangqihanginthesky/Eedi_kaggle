# Copasta's part
In this part, we will create a retrieval model.

## 1. Data Preparation
Download the retrieve-train-pipeline2 data from [Google Drive](../../../../data/retrieve_train/README.md) and place it in the input directory.

## 2. Training
### Environment
Lambda Cloud

*  1x H100 (80 GB PCIe) Instance

Run the following pip command:

```
pip install -r requirements.txt
```

### Notebook
Please run the following notebook. Please adjust the I/O directory according to your environment.

* step1_training_retrieval.ipynb

## 3. Model merge
### Environment
Google Colaboratory

* A100 (40 GB)

### Notebook
Please run the following notebook. Please adjust the I/O directory according to your environment.

* step2_merge_llm.ipynb

## 4. Quantization
### Environment
Google Colaboratory

* A100 (40 GB)

### Notebook
Please run the following notebook. Please adjust the I/O directory according to your environment.

* step3_quant_llm.ipynb
