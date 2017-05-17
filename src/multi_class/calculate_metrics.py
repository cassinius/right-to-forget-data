import sklearn.metrics as skmetrics

def calculateMetrics(predictions, truth):
    precision = skmetrics.precision_score(predictions, truth, average="weighted")
    recall = skmetrics.recall_score(predictions, truth, average="weighted")
    f1 = (2.0 * precision * recall) / (precision + recall)
    accuracy = skmetrics.accuracy_score(predictions, truth, normalize=True, sample_weight=None)
    return precision, recall, f1, accuracy