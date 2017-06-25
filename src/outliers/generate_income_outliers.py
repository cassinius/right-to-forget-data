import numpy as np
from sklearn.ensemble import IsolationForest
from src.multi_class import input_preproc

import pandas as pd


OUTPUT_DIR = "../../data/outliers/adults_target_income/outliers/"
OUTPUT_NAME_BULK = "adults_outliers_removed_"
INPUT_FILE = "../../data/anonymization/adults_target_income/original/adults_original_dataset.csv"
INPUT_COLS = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "marital-status",
        "relationship",
        "occupation",
        "income"
     ]
TARGET_COL = "income"

CONTAMINATION_LEVELS = np.linspace(0.1, 0.95, 18)

rng = np.random.RandomState(42)


# We need the original data as well
def getOriginalData():
    return input_preproc.returnOriginalDataset(
        INPUT_FILE,
        INPUT_COLS,
        TARGET_COL
    )


def getEncodedData():
    return input_preproc.readFromDataset(
        INPUT_FILE,
        INPUT_COLS,
        TARGET_COL
    )


def calculateOutliersIndicesAtLevel(encoded_data, level):
    clf = IsolationForest(n_estimators=100,
                          max_samples='auto',
                          contamination=contamination_level,
                          # max_features=1,
                          bootstrap=False,
                          n_jobs=-1,
                          random_state=rng)

    # Split into predictors and target
    X_adults = np.array( encoded_data[encoded_data.columns.difference([TARGET_COL])] )

    clf.fit(X_adults)
    y_pred_adults = clf.predict(X_adults)
    indices = np.where(y_pred_adults == -1)[0]

    # print("%d Outliers found at contamination level %.2f" %(len(indices), level))

    # print "\n========================\n"
    # print("%d Outliers found, %s" %(len(indices), "at indices:"))
    # print indices
    # print "\n========================\n"

    return indices


# Just for testing, outsource ASAP
def outputCleanedFile(cleaned_data, contamination_level):
    file_name = OUTPUT_DIR + OUTPUT_NAME_BULK + str(contamination_level) + ".csv"
    cleaned_data.to_csv(file_name,
                        header=INPUT_COLS,
                        index=False,
                        encoding="utf-8")


if __name__ == "__main__":
    original_data = getOriginalData()
    encoded_data = getEncodedData()
    print("Std. Deviation ORIGINAL: %s" % (np.std(np.array(encoded_data[encoded_data.columns.difference([TARGET_COL])]))))

    for contamination_level in CONTAMINATION_LEVELS:
        outlier_indices = calculateOutliersIndicesAtLevel(encoded_data, contamination_level)
        cleaned_data = original_data.drop(original_data.index[outlier_indices])

        cleaned_encoded = encoded_data.drop(encoded_data.index[outlier_indices])
        double_clean = cleaned_encoded[cleaned_encoded.columns.difference([TARGET_COL])]
        print("Std. Deviation PURIFIED: %s" % (np.std(np.array(double_clean))))

        # print("Length of remaining data: %d \n" % (len(cleaned_data)))
        outputCleanedFile(cleaned_data, contamination_level)