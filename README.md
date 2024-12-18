# About this project
## Folder structure
```
Solution
├─ README.md                    # Overall introduction to the project: methodology, architecture, dependencies, and how to run
├─ LICENSE                      # License information (optional)
├─ requirements.txt             # Required dependencies or environment.yaml


├─ docs/
│  ├─ methodology.md            # Detailed explanation of the entire approach (from Preprocessing to Reranking)
│  ├─ synthetic_data.md         # In-depth explanation of synthetic data generation and parameters
│  ├─ retriever_training.md     # Details on retriever model training, architecture, and parameters
│  ├─ reranker_training.md      # Detailed steps for training the reranker models
│  ├─ chain_of_thought.md       # Explanation of Chain-of-Thought generation and its performance impact
│  ├─ misconception_aug.md      # Misconception augmentation process and example prompts
│  ├─ quantization.md           # Model quantization process and inference optimization details
│  ├─ inference_guide.md        # Instructions for inference, including vLLM usage and TTA strategies
│  ├─ references.md             # References and related resources
│  └─ experiments/
│     ├─ ablation_study.md     # Records of ablation experiments
│     ├─ performance_tables.md # Performance tables for Private and Public Leaderboards
│     └─ notes.md              # Additional experimental notes and useful tips


├─ data/
│  ├─ raw/                      # Original data (e.g., train.csv, test.csv, subject_metadata.csv)
│  ├─ processed/                # Preprocessed data with parent subject info integrated
│  ├─ synthetic_generation1/    # Synthetic data from Generation1
│  ├─ synthetic_generation2/    # Synthetic data from Generation2
│  ├─ synthetic_generation3/    # Synthetic data from Generation3
│  ├─ misconceptions/           # Misconception lists and extended explanation texts
│  ├─ cot_explanations/         # Chain-of-thought explanation data
│  └─ calibration_data/         # Data samples used for quantization calibration


├─ src/
│  ├─ preprocessing/
│  │  └─ preprocess.py          # Scripts for preprocessing data: integrating parent subjects, cleaning
│  │
│  ├─ synthetic_data_generation/
│  │  ├─ generation1.py         # Code to generate Generation1 synthetic data
│  │  ├─ generation2.py         # Code to generate Generation2 synthetic data
│  │  ├─ generation3.py         # Code to generate Generation3 synthetic data
│  │  ├─ filters.py             # Scripts for quality filtering using gpt-4o-mini
│  │  └─ misconception_expansion.py # Scripts for augmenting misconception descriptions using LLMs
│
│  ├─ retrieval/
│  │  ├─ train_retriever.py     # Training scripts for the retriever models (Pipeline1 & Pipeline2)
│  │  ├─ inference_retriever.py # Inference scripts for the retriever
│  │  ├─ model_configs/         # Configuration files for each retriever model (e.g., Linq, Qwen2.5)
│  │  └─ utils.py               # Utility functions (e.g., for pooling methods)
│
│  ├─ reranking/
│  │  ├─ train_reranker.py      # Training scripts for the reranker models (QLoRA training, listwise approach)
│  │  ├─ inference_reranker.py  # Reranking inference scripts (sliding window, TTA)
│  │  ├─ model_configs/         # Configuration files for each reranker model (e.g., Qwen2.5-72B, Llama-70B)
│  │  ├─ utils.py               # Utility functions for reranking
│  │  └─ quantization.py        # Scripts for auto-round quantization
│
│  ├─ chain_of_thought/
│  │  └─ generate_cot.py        # Scripts to generate Chain-of-Thought using qwen2.5-32B for training/inference
│
│  └─ evaluation/
│     └─ evaluate.py            # Evaluation scripts for retriever and reranker outputs


├─ experiments/
│  ├─ notebooks/                # Jupyter Notebooks for demonstration, visualization, and experiment tracking
│  ├─ logs/                     # Logs generated during training and inference
│  ├─ results/                  # Evaluation metrics, LB submission files, and final results
│  └─ configs/                  # Configuration files for experiments (e.g., hyperparameters, data splits)


└─ scripts/
  ├─ run_full_pipeline.sh      # One-click script to run the entire pipeline from data preprocessing to final reranking
  ├─ run_retrieval.sh          # Script to run the retrieval stage only
  ├─ run_rerank.sh             # Script to run the reranking stage only
  └─ run_evaluation.sh         # Script to run evaluation on the produced outputs
```

## How to run
