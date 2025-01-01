This is the code for the reranking model used in our 2nd-place solution for the [Kaggle EEDI competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics).

## Hardware
* GPU: 1x Nvidia H100 SXM 80GB
* CPU: AMD EPYC 9124 32 vCPUs
* CPU Memory: 256 GB

## Environment setup
We use [uv](https://docs.astral.sh/uv/) for the Python environment setup. After installing uv, up the environment with

```
cd src/4.reranking
uv sync

# FlashAttention needs to be installed separately
uv pip install flash-attn --no-build-isolation

# activate environment
. .venv/bin/activate
```

## Data preparation
Please generate [synthetic data](https://github.com/wangqihanginthesky/Eedi_kaggle/tree/rihanpiggy/data/2.synthetic_data_generation) in advance. Alternatively, the required synthetic data can be downloaded from the sources listed below.

https://github.com/wangqihanginthesky/Eedi_kaggle/tree/main/data/retrieve_train

Next, place the competition data and synthetic data in the directories specified in `config.yaml`.

## Training QLoRA models
Please run the following Jupyter notebooks depending on the model:
- Qwen2.5-14B: `exp_qwen2.5_14B_train.ipynb`
- Qwen2.5-72B: `exp_qwen2.5_72B_train.ipynb`
- Llama3.3-70B: `exp_llama3.3_70B_train.ipynb`

## Post training quantization
Run the following command.
- Qwen2.5-14B: `python quantize.py qwne25_14B_fold0/merged_model/`
- Qwen2.5-72B: `python quantize.py qwne25_72B_fold0/merged_model/`
- Llama3.3-70B: `python quantize.py llama33_70B_fold0/merged_model/`

After quantization is complete, the quantized model used for submission will be output under `merged_model_quant` in each directory.
