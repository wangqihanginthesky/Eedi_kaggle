# About this project
This is a solution for the Eedi Kaggle competition focusing on student misconception detection.

## Folder structure
```
.
├── README.md                          # Project documentation
├── Solution.md                        # Detailed solution explanation
├── data/                              # Data directory
│   ├── calibration_data               # Model calibration datasets
│   ├── cot_explanations               # Chain of thought explanation data
│   ├── misconceptions                 # Misconception examples
│   ├── processed                      # Processed datasets
│   ├── raw                            # Raw competition data
│   └── synthetic_generation           # Synthetic dataset
├── docs/                              # Documentation files
│   ├── chain_of_thought.md            # Chain of thought approach details
│   ├── experiments                    # Experiment results and analysis
│   ├── inference_guide.md             # Guide for model inference
│   ├── methodology.md                 # Overall methodology
│   ├── misconception_aug.md           # Misconception augmentation details
│   ├── quantization.md                # Model quantization guide
│   ├── references.md                  # Reference papers and resources
│   ├── reranker_training.md           # Reranker model training guide
│   ├── retriever_training.md          # Retriever model training guide
│   └── synthetic_data.md              # Synthetic data generation details
├── requirements.txt                   # Python dependencies
├── scripts/                           # Shell scripts for running components
│   ├── prepare_data.py                # Prepare data for train and infer
│   ├── train.py                       # Train Retrieval and Reranker
│   └── predict.py                     # Predict for final result
└── src/                               # Source code
    ├── chain_of_thought               # Chain of thought implementation
    ├── evaluation                     # Evaluation metrics and tools
    ├── preprocessing                  # Data preprocessing
    ├── reranking                      # Reranking model
    ├── retrieval                      # Retrieval model
    └── synthetic_data_generation      # Synthetic data generation code

```

## How to run

### 1. Parepare data: 
If you want to skip the synthetic data generation (render) and use pre-generated files instead, follow these steps:

Install Dependencies
As before, install all required Python libraries by running:

```
pip install -r requirements.txt
```

Check the Configuration
Ensure your pre-rendered synthetic data files are located in the correct directories:
```
data/synthetic_generation
```
Update paths in SETTINGS.json or directly verify the expected locations in prepare_data.py.

Run the Script with Skip Render
Use the --skip_render flag to skip synthetic data generation and load pre-rendered files:

```
python scripts/prepare_data.py --skip_render
```
This script will:

- Load the raw competition data from data/raw.
- Skip generating synthetic data and instead read from pre-rendered files.
- Perform preprocessing on all datasets and combine them with synthetic data.
- Save the processed datasets into data/processed.
- Verify Outputs
- Check the data/processed directory for the final combined dataset and ensure all required files (e.g., training and validation data) are ready.

This mode is useful if synthetic data has already been generated and validated, saving time during preprocessing. Make sure the pre-rendered files are complete and compatible with the pipeline.


You can generate the data by running the following command:
```
python scripts/prepare_data.py
```
This will prepare the required data files in the expected format, making them ready for training.