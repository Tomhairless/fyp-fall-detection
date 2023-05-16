import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

df = pd.read_csv('csv_0304/more_feature_combined.csv')

selected_features = ['centroid_h', 'ratio', 'down_velocity']

features = pd.read_csv('csv_0304/more_feature_combined.csv',
                       usecols=selected_features)
labels = df.label_2
print(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.40,
                                                                            random_state=0, shuffle=False)
print(test_features)

# CART decision tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=10, min_samples_split=10)

# # Calculate the pruning path
# path = clf.cost_complexity_pruning_path(features, labels)
#
# # Extract the ccp_alphas and impurities from the path
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
# # Print the ccp_alphas and impurities
# print("ccp_alphas:", ccp_alphas)
# print("impurities:", impurities)
# print(ccp_alphas / impurities)

clf = clf.fit(train_features, train_labels)

test_predict = clf.predict(test_features)

accuracy_results = classification_report(test_labels, test_predict)
print(accuracy_results)

print(tree.export_text(clf, feature_names=selected_features))
dot_data = export_graphviz(clf)
graph = graphviz.Source(dot_data)

graph.view()

print(clf.feature_importances_)

eval = pd.read_csv('csv_0304/eval.csv', usecols=selected_features)
print(type(eval))
print(eval)
results = clf.predict(eval)
print(np.array2string(results, separator=','))
print(results.shape)

# rf = RandomForestClassifier(n_estimators=140, max_depth=12)
# rf.fit(train_features, train_labels)
#
# test_predict = rf.predict(test_features)
# score = accuracy_score(test_labels, test_predict)
#
# accuracy = accuracy_score(test_predict, test_labels)
# print("Accuracy:", accuracy)

# Create the confusion matrix
cm = confusion_matrix(test_labels, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()
