# Create VectorDB using subject_metadata.csv
```
python3 run create_vector_db.py
```

# Preprocess(train)
```
python3 run_subject_search.py --filepath ./train.csv --index-db-path ./faiss_subject.index --subject-master-path ./subject_metadata.csv --model-path Alibaba-NLP/gte-base-en-v1.5 --output-dir .

python3 run_prepare_df.py --filepath ./train.csv --output-dir .
```

# Preprocess(test)
```
python3 run_subject_search.py --filepath ./test.csv --index-db-path ./faiss_subject.index --subject-master-path ./subject_metadata.csv --model-path Alibaba-NLP/gte-base-en-v1.5 --output-dir .

python3 run_prepare_df.py --filepath ./test.csv --output-dir .
```
