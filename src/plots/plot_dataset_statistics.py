'''
    We want to implement:
    1. Target column values
    2. Frequency of 1.
    3. Skewedness of dataset as measured by 2.
    ...
'''


from src.multi_class import input_preproc

MODE = 'anonymization'
OUTLIER_TARGET = 'original/'

CONFIG_INCOME = {
    'TARGET': "../../data/" + MODE + "/adults_target_income/" + OUTLIER_TARGET,
    'OUTPUT': "../../output/" + MODE + "/adults_target_income/" + OUTLIER_TARGET,
    'INPUT_COLS': [
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
     ],
    'TARGET_COL': "income"
}

config = CONFIG_INCOME

input_file = 'adults_original_dataset.csv'

encoded_data = input_preproc.readFromDataset(
                    config['TARGET'] + input_file,
                    config['INPUT_COLS'],
                    config['TARGET_COL']
                )

target = encoded_data[config['TARGET_COL']]
# print target
# print type(target)

values = target.value_counts(  normalize=False,
                               sort=True,
                               ascending=False,
                               bins=None,
                               dropna=True)

# values = encoded_data.count()

print "Absolute values:"
print values
# print sum(values)

print "Fractions:"
for key, value in values.iteritems():
    print("%s : %.2f" %(key, float(value)/sum(values)))