import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump


# Split the data into features (X) and target (y)
df = pd.read_csv('csv_0304/final_decision_0320.csv')

features_selection = ['ratio', 'angle', 'down_velocity', 'centroid_h', 'contour_w']

X = pd.read_csv('csv_0304/final_decision_0320.csv',
                usecols=features_selection)
print(X)
y = df.label_fall

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)

rf = RandomForestClassifier(n_estimators=120, max_features=3)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_results = classification_report(y_pred=y_pred, y_true=y_test)
print(accuracy_results)

# # Define parameter grid for grid search
# param_grid = {'n_estimators': [50, 100, 150, 200],
#               'max_depth': [8, 10, 12, 14],
#               'min_samples_split': [2, 4, 6],
#               'min_samples_leaf': [1, 2, 4]}
#
# # Create Random Forest classifier object
# rf_clf = RandomForestClassifier(random_state=42)
#
# # Perform grid search using 5-fold cross-validation
# grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid)
#
# # Fit the grid search object to the training data
# grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters and corresponding accuracy score
# print("Best hyperparameters: ", grid_search.best_params_)
# print("Best accuracy score: ", grid_search.best_score_)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf.classes_)
disp.plot()
plt.show()


# Export the trained model to a file
print(rf)
dump(rf, 'model/TEST.joblib')

# eval = pd.read_csv('csv_0304/eval.csv', usecols=features_selection)
# results = rf.predict(eval)
# print(np.array2string(results, separator=','))
# print(results.shape)
