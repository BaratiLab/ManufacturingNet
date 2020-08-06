from ManufacturingNet.shallow_learning_methods import LogRegression, RandomForest, SVM, XGBoost
import pandas as pd

# Prep data
data = pd.read_csv("classification_test_data.csv")
data = data.to_numpy()
atts = data[0:, 0:13]
labels = data[0:, 13]

# Instantiate models
log_reg = LogRegression(atts, labels)
random_forest = RandomForest(atts, labels)
svm = SVM(atts, labels)
xgb = XGBoost(atts, labels)
failures = []

# Test models
try:
    log_reg.run()

    if log_reg.get_accuracy() is None:
        raise Exception()
    
    print("LogRegression ran successfully.")
except Exception as e:
    print("LogRegression failed.")
    print(e)
    failures.append("LogRegression")

print()

try:
    random_forest.run_classifier()

    if random_forest.get_accuracy() is None:
        raise Exception()
    
    print("RandomForest ran successfully.")
except Exception as e:
    print("RandomForest failed.")
    print(e)
    failures.append("RandomForest")

print()

try:
    svm.run_SVC()

    if svm.get_accuracy_SVC() is None:
        raise Exception()
    
    print("SVC ran successfully.")
except Exception as e:
    print("SVC failed.")
    print(e)
    failures.append("SVC")

print()

try:
    svm.run_nu_SVC()

    if svm.get_accuracy_nu_SVC() is None:
        raise Exception()
    
    print("NuSVC ran successfully.")
except Exception as e:
    print("NuSVC failed.")
    print(e)
    failures.append("NuSVC")

print()

try:
    svm.run_linear_SVC()

    if svm.get_accuracy_linear_SVC() is None:
        raise Exception()
    
    print("LinearSVC ran successfully.")
except Exception as e:
    print("LinearSVC failed.")
    print(e)
    failures.append("LinearSVC")

print()

try:
    xgb.run_classifier()

    if xgb.get_accuracy() is None:
        raise Exception()
    
    print("XGBoost ran successfully.")
except Exception as e:
    print("XGBoost failed.")
    print(e)
    failures.append("XGBoost")

print()

# Print failures
print("Failures:", failures)