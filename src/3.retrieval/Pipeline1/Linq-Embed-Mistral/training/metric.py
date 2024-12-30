import numpy as np

# https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code

class KaggleMetric(object):
    @staticmethod
    def apk(actual, predicted, k=25):
        """
        Computes the average precision at k.
        
        This function computes the average prescision at k between two lists of
        items.
        
        Parameters
        ----------
        actual : list
                A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
            
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        
        if not actual:
            return 0.0

        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            # first condition checks whether it is valid prediction
            # second condition checks if prediction is not repeated
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        return score / min(len(actual), k)
    
    @staticmethod
    def mapk(actual, predicted, k=25):
        """
        Computes the mean average precision at k.
        
        This function computes the mean average prescision at k between two lists
        of lists of items.
        
        Parameters
        ----------
        actual : list
                A list of lists of elements that are to be predicted 
                (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
            
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """
        
        return np.mean([KaggleMetric.apk(a,p,k) for a,p in zip(actual, predicted)])

    @staticmethod
    def recall_at_k(y_true, y_pred, k):
        """
        Recall@kを計算する関数

        Args:
        y_true (list of lists): 各クエリに対する正解のリスト
        y_pred (list of lists): 各クエリに対する予測のリスト
        k (int): 上位k件の予測を評価対象とする

        Returns:
        float: Recall@kの平均値
        """
        recalls = []
        for true_items, pred_items in zip(y_true, y_pred):
            if len(pred_items) > k:
                pred_items = pred_items[:k]
            hits = len(set(true_items) & set(pred_items))
            recalls.append(hits / len(true_items))
        return np.mean(recalls)