from ManufacturingNet.shallow_learning_methods import LinRegression, RandomForest, SVM, XGBoost
import pandas as pd

# Prep data
data = pd.read_csv("regression_test_data.csv")
data = data.to_numpy()
atts = data[0:,2:6]
labels = data[0:,6]

# Instantiate models
lin_reg = LinRegression(atts, labels)
random_forest = RandomForest(atts, labels)
svm = SVM(atts, labels)
xgb = XGBoost(atts, labels)
failures = []

# Test models
try:
    lin_reg.run()

    if lin_reg.get_r2_score() is None:
        raise Exception()
    
    print("LinRegression ran successfully.")
    print("r2 score:", lin_reg.get_r2_score())
except Exception as e:
    print("LinRegression failed.")
    print(e)
    failures.append("LinRegression")

print()

try:
    random_forest.run_regressor()

    if random_forest.get_r2_score() is None:
        raise Exception()
    
    print("RandomForest ran successfully.")
    print("r2 score:", random_forest.get_r2_score())
except Exception as e:
    print("RandomForest failed.")
    print(e)
    failures.append("RandomForest")

print()

try:
    svm.run_SVR()

    if svm.get_r2_score_SVR() is None:
        raise Exception()
    
    print("SVR ran successfully.")
    print("r2 score:", svm.get_r2_score_SVR())
except Exception as e:
    print("SVR failed.")
    print(e)
    failures.append("SVR")

print()

try:
    svm.run_nu_SVR()

    if svm.get_r2_score_nu_SVR() is None:
        raise Exception()
    
    print("NuSVR ran successfully.")
    print("r2 score:", svm.get_r2_score_nu_SVR())
except Exception as e:
    print("NuSVR failed.")
    print(e)
    failures.append("NuSVR")

print()

try:
    svm.run_linear_SVR()

    if svm.get_r2_score_linear_SVR() is None:
        raise Exception()
    
    print("LinearSVR ran successfully.")
    print("r2 score:", svm.get_r2_score_linear_SVR())
except Exception as e:
    print("LinearSVR failed.")
    print(e)
    failures.append("LinearSVR")

print()

try:
    xgb.run_regressor()

    if xgb.get_r2_score() is None:
        raise Exception()
    
    print("XGBoost ran successfully.")
    print("r2 score:", xgb.get_r2_score())
except Exception as e:
    print("XGBoost failed.")
    print(e)
    failures.append("XGBoost")

print()

# Print failures
print("Failures:", failures)