# run training
```
CUDA_VISIBLE_DEVICES=0 python3 run_v3_full_train.py --config config/e032-mistral-synthetic-gen1-gen2-gen3.yaml --filepath ../../../../data/retrieve_train/train_5folds_with_synthetic_gen1_gen2_thre3.csv --filepath-misconception ../../../../data/retrieve_train/misconception_mapping.csv --fold 100 --output-dir ./ckpt_e032
```
