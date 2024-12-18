# About this project
This is a solution for the Eedi Kaggle competition focusing on student misconception detection.

## Folder structure
```
.
├── README.md                        documentation
├── Solution.md                        # Detailed solution explanation
├── data/                             # Data directory
│   ├── calibration_data              # Model calibration datasets
│   ├── cot_explanations             # Chain of thought explanation data
│   ├── misconceptions               # Misconception examples
│   ├── processed                    # Processed datasets
│   ├── raw                         # Raw competition data
│   ├── synthetic_generation1        # First synthetic dataset
│   ├── synthetic_generation2        # Second synthetic dataset
│   └── synthetic_generation3        # Third synthetic dataset
├── docs/                            # Documentation files
│   ├── chain_of_thought.md         # Chain of thought approach details
│   ├── experiments                 # Experiment results and analysis
│   ├── inference_guide.md          # Guide for model inference
│   ├── methodology.md              # Overall methodology
│   ├── misconception_aug.md        # Misconception augmentation details
│   ├── quantization.md            # Model quantization guide
│   ├── references.md              # Reference papers and resources
│   ├── reranker_training.md       # Reranker model training guide
│   ├── retriever_training.md      # Retriever model training guide
│   └── synthetic_data.md          # Synthetic data generation details
├── requirements.txt                # Python dependencies
├── scripts/                       # Shell scripts for running components
│   ├── run_evaluation.sh         # Evaluation script
│   ├── run_full_pipeline.sh      # End-to-end pipeline script
│   ├── run_rerank.sh            # Reranking script
│   └── run_retrieval.sh         # Retrieval script
└── src/                         # Source code
    ├── chain_of_thought        # Chain of thought implementation
    ├── evaluation             # Evaluation metrics and tools
    ├── preprocessing          # Data preprocessing
    ├── reranking             # Reranking model
    ├── retrieval             # Retrieval model
    └── synthetic_data_generation  # Synthetic data generation code
```

## How to run
