import numpy as np

class SemanticSearcher(object):
    @staticmethod
    def _cosine_similarity(embeddings, embeddings_mapping, batch_size=None):
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_mapping = embeddings_mapping / np.linalg.norm(embeddings_mapping, axis=1, keepdims=True)
        if batch_size is not None:
            cosine_similarity = []
            for i in range(0, len(embeddings), batch_size):
                cosine_similarity.append(np.dot(embeddings[i:i+batch_size], embeddings_mapping.T))
            cosine_similarity = np.concatenate(cosine_similarity, axis=0)
        else:
            cosine_similarity = np.dot(embeddings, embeddings_mapping.T)
        return cosine_similarity

    @staticmethod
    def cosine_similarity_search(embeddings, embeddings_mapping, ids, batch_size=None):
        cosine_similarity = SemanticSearcher._cosine_similarity(embeddings, embeddings_mapping, batch_size)
        
        predict_ids = []
        for i in range(len(cosine_similarity)):
            descending_idx = np.argsort(cosine_similarity[i])[::-1]
            predict_ids.append(ids[descending_idx].tolist())

        return predict_ids

    @staticmethod
    def rerank(df_test, scores, top_k):
        rerank_ids = []
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            predict_ids = np.array(row['predict_ids'])
            score = scores[i*top_k:(i+1)*top_k]
            descending_idx = np.argsort(score)[::-1]
            rerank_ids.append(predict_ids[:top_k][descending_idx].tolist() + predict_ids[top_k:].tolist())
        return rerank_ids
