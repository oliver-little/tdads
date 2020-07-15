# TDADS config

models = {
    "zero_r": {
        "fit": None,
        "predict": None,
    },
    "feedforward": {
        "fit": {
            "plot": False,
        },
        "predict": {
            "plot": False,
        },
        "enabled": False,
    },
    "rnn": {
        "fit": {
            "epochs": 594
        },
        "predict": None,
        "enabled": False,
    },
    "linear_regression": {
        "fit": None,
        "predict": None,
        "enabled": True,
    },
    "logistic_regression": {
        "fit": None,
        "predict": None,
        "enabled": True,
    },
    "feature_detect": {
        "fit": None,
        "predict": None,
        "enabled": True,
    },
    "naive_bayes": {
        "fit": {

        },
        "predict": {

        },
        "enabled": True,
    },
    "svm": {
        "fit": {

        },
        "predict": {

        },
        "enabled": True,
    },
    "random_forest": {
        "fit": {

        },
        "predict": {

        },
        "enabled": True,
    },
    "kmeans": {
        "fit": {
            "plot": False
        },
        "predict": {

        },
        "enabled": True,
    },
    "cnn": {
        "fit": {
            "nets": 15,
            "epochs": 45,
            "plot": False
        },
        "predict": {

        },
        "enabled": True,
    },
    "kneighbours": {
        "fit": None,
        "predict": {
            "threads_n": 8,
            "k_value": 10
        },
        "enabled": True,
    },
}
