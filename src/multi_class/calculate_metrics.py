import sklearn.metrics as skmetrics

def calculateMetrics(predictions, truth):
    precision = skmetrics.precision_score(predictions, truth, average="macro")
    recall = skmetrics.recall_score(predictions, truth, average="macro")
    f1 = (2.0 * precision * recall) / (precision + recall)
    return precision, recall, f1