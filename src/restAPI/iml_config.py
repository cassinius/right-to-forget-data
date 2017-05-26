CROSS_VALIDATION_K = 5


INPUT_COLS = {
    "education-num": [
        "age",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "marital-status",
        "relationship",
        "occupation",
        "income",
        "education-num"
    ],
    "marital-status": [
        "age",
        "education-num",
        "hours-per-week",
        "workclass",
        "native-country",
        "sex",
        "race",
        "relationship",
        "occupation",
        "income",
        "marital-status"
    ],
    "income": [
        "age",
        "education-num",
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
}


ALGORITHMS = [
    'linear_svc',
    'logistic_regression',
    'gradient_boosting',
    'random_forest',
]