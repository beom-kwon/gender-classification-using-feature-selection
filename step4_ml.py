import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_halving_search_cv  # noqa (no quality assurance)
from sklearn.model_selection import HalvingGridSearchCV

import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)


def hyperparameter_optimization(name, clf, x_train, y_train):
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, clf)
    alg_name = pipeline.steps[1][0]
    if name == "qda":  # Quadratic Discriminant Analysis
        param_grid = {"{}__store_covariance".format(alg_name): [True, False],
                      "{}__tol".format(alg_name): [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]}
    elif name == "gpc":  # Gaussian Process Classifier
        param_grid = {"{}__warm_start".format(alg_name): [True, False],
                      "{}__copy_X_train".format(alg_name): [True, False]}
    elif name == "lr":  # Logistic Regression
        param_grid = {"{}__penalty".format(alg_name): ["l1", "l2", "elasticnet", None],
                      "{}__dual".format(alg_name): [True, False],
                      "{}__tol".format(alg_name): [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
                      "{}__C".format(alg_name): [1e-2, 1e-1, 1e-0, 1e-1, 1e-2],
                      "{}__fit_intercept".format(alg_name): [True, False],
                      "{}__warm_start".format(alg_name): [True, False]}
    elif name == "gnb":  # Gaussian Naive Bayes
        param_grid = {"{}__var_smoothing".format(alg_name): [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    elif name == "knn":  # k-Nearest Neighbors
        param_grid = {"{}__n_neighbors".format(alg_name): np.arange(1, 11, 2),
                      "{}__weights".format(alg_name): ["uniform", "distance"],
                      "{}__algorithm".format(alg_name): ["auto"],
                      "{}__leaf_size".format(alg_name): [10, 20, 30, 40, 50],
                      "{}__metric".format(alg_name): ["minkowski", "euclidean", "cityblock"]}
    elif name == "dt":  # Decision Tree
        param_grid = {"{}__criterion".format(alg_name): ["gini", "entropy", "log_loss"],
                      "{}__splitter".format(alg_name): ["best", "random"],
                      "{}__max_depth".format(alg_name): [3, 4, 5, 6, 7, 8, 9, 10]}
    elif name == "svm":  # Support Vector Machine
        param_grid = {"{}__C".format(alg_name): [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                      "{}__kernel".format(alg_name): ["linear", "poly", "rbf", "sigmoid"],
                      "{}__degree".format(alg_name): [1, 2, 3, 4, 5],
                      "{}__gamma".format(alg_name): ["scale", "auto"],
                      "{}__coef0".format(alg_name): [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                      '{}__shrinking'.format(alg_name): [True, False]}

    hgs = HalvingGridSearchCV(pipeline,
                              param_grid=param_grid,
                              factor=3,     # the Proportion of Candidates Selected for Each Subsequent Iteration
                              cv=4,         # 4-Fold Cross Validation
                              random_state=42,
                              refit=True)   # If True, Refit an Estimator Using the Best Parameters

    hgs.fit(x_train, y_train)
    return hgs


if __name__ == "__main__":
    dataset = np.load("./feature.npz")

    person_id = dataset["person_id"]
    x = dataset['x']
    y = dataset['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)

    # Two Control Parameters
    N = 40      # The Number of Selected Features
    c_type = 6  # Classifier Type

    train_data = np.column_stack((x_train, y_train))
    df = pd.DataFrame(train_data)
    abs_corr = np.absolute(df.corr().iloc[:-1, 82])
    idx = np.argpartition(abs_corr, -N)[-N:]

    x_train = df[idx].to_numpy()
    x_test = pd.DataFrame(x_test)[idx].to_numpy()

    if c_type == 0:
        name = "qda"
    elif c_type == 1:
        name = "gpc"
    elif c_type == 2:
        name = "lr"
    elif c_type == 3:
        name = "gnb"
    elif c_type == 4:
        name = "knn"
    elif c_type == 5:
        name = "dt"
    elif c_type == 6:
        name = "svm"

    clf = {"qda": QuadraticDiscriminantAnalysis(), # Quadratic Discriminant Analysis
           "gpc": GaussianProcessClassifier(),     # Gaussian Process Classifier
           "lr": LogisticRegression(),             # Logistic Regression
           "gnb": GaussianNB(),                    # Gaussian Naive Bayes
           "knn": KNeighborsClassifier(),          # k-Nearest Neighbors
           "dt": DecisionTreeClassifier(),         # Decision Tree
           "svm": SVC()                            # Support Vector Machine
           }
    print(">> Classifier Type:", name)

    model = hyperparameter_optimization(name, clf[name], x_train, y_train)
    y_pred = model.predict(x_test)

    tp, fn, fp, tn = confusion_matrix(y_test, y_pred, labels=model.classes_).ravel()
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Female", "Male"], cmap=plt.cm.Blues)

    print(">> ACC: %.4f" % ((tp + tn) / (tp + fn + fp + tn)))
    print(">> TPR: %.4f" % (tp / (tp + fn)))
    print(">> TNR: %.4f" % (tn / (tn + fp)))

    # print(model.best_params_)