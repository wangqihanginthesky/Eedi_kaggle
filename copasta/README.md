# Copasta's part
In this part, we will create a retrieval model.

## [WIP] Data Preparation
* 洪さんパートの構成に対応して更新
* 基本的な流れの想定
    * fold datasetの作成・配置
    * 合成データの作成・配置

## Training
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

## Model merge
### Environment
Google Colaboratory

* A100 (40 GB)

### Notebook
Please run the following notebook. Please adjust the I/O directory according to your environment.

* step2_merge_llm.ipynb

## Quantization
### Environment
Google Colaboratory

* A100 (40 GB)

### Notebook
Please run the following notebook. Please adjust the I/O directory according to your environment.

* step3_quant_llm.ipynb
