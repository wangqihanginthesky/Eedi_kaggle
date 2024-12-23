# Create VectorDB using subject_metadata.csv
```
python3 run create_vector_db.py
```

# Preprocess(train)
```
python3 run_subject_search.py --filepath ../../data/raw/train.csv --index-db-path ./faiss_subject.index --subject-master-path ./subject_metadata.csv --model-path Alibaba-NLP/gte-base-en-v1.5 --output-dir ../../data/0.preprocessing

python3 run_prepare_df.py --filepath ../../data/0.preprocessing/train.csv --output-dir ../../data/0.preprocessing
```
