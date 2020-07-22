from ManufacturingNet.shallow_learning_methods import LogRegression, MLP, RandomForest, SVM, XGBoost
import pandas as pd

# Prep data
data = pd.read_csv("classification_test_data.csv")
data = data.to_numpy()
atts = data[0:, 0:13]
labels = data[0:, 13]

# Instantiate models
log_reg = LogRegression(atts, labels)
mlp = MLP(atts, labels)
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
    print("accuracy:", log_reg.get_accuracy())
except Exception as e:
    print("LogRegression failed.")
    print(e)
    failures.append("LogRegression")

print()

try:
    mlp.run()
    
    if mlp.get_accuracy() is None:
        raise Exception()
    
    print("MLP ran successfully.")
    print("accuracy:", mlp.get_accuracy())
except Exception as e:
    print("MLP failed.")
    print(e)
    failures.append("MLP")

print()

try:
    random_forest.run_classifier()

    if random_forest.get_accuracy() is None:
        raise Exception()
    
    print("RandomForest ran successfully.")
    print("accuracy:", random_forest.get_accuracy())
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
    print("accuracy:", svm.get_accuracy_SVC())
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
    print("accuracy:", svm.get_accuracy_nu_SVC())
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
    print("accuracy:", svm.get_accuracy_linear_SVC())
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
    print("accuracy:", xgb.get_accuracy())
except Exception as e:
    print("XGBoost failed.")
    print(e)
    failures.append("XGBoost")

print()

# Print failures
print("Failures:", failures)