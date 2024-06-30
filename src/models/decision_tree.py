import joblib
import utility
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------- Initialize data -------
size = 0.3
xtrain_TREE, xvalid_TREE, ytrain_TREE, yvalid_TREE = train_test_split(X_train_DF, Y_train_DF, test_size=size, random_state=42)

# ------- Entropy -------
model_tree = DecisionTreeClassifier(max_depth=None, criterion='entropy', random_state=42)
model_tree.fit(xtrain_TREE, ytrain_TREE)
print(f'Decision Tree Model (Entropy): {model_tree}')


# ------- Print tree -------
plt.figure(figsize=(16, 9))
plot_tree(model_tree, filled=True, feature_names=xtrain_TREE.columns, class_names=True)
plt.title("Decision Tree - Entropy")
plt.show()


# ------- Print accuracy train, validation -------
utility.print_accuricy_tree("train", 'None', xtrain_TREE, ytrain_TREE, model_tree, size)
utility.print_accuricy_tree("validation", 'None', xvalid_TREE, yvalid_TREE, model_tree, size)


# ------- Grid search -------
hyperparameters = {
    'max_depth': np.arange(1, 20),
    'criterion': ['entropy', 'gini'],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=hyperparameters, cv=10)
grid_search.fit(xtrain_TREE, ytrain_TREE)
best_model_tree = grid_search.best_estimator_
print('Best Model after Grid Search:')
print(best_model_tree)
print('Best Parameters:')
print(grid_search.best_params_)



# ------- Plot best model tree -------
plt.figure(figsize=(16, 9))
plot_tree(best_model_tree, filled=True, feature_names=xtrain_TREE.columns, class_names=True)
plt.title(f"Best Decision Tree - Depth: {grid_search.best_params_['max_depth']}")
plt.show()


# ------- Train and evaluate best model -------
best_model_tree.fit(xtrain_TREE, ytrain_TREE)
utility.print_accuracy_tree("Train", grid_search.best_params_['max_depth'], xtrain_TREE, ytrain_TREE, best_model_tree, size)
utility.print_accuracy_tree("Validation", grid_search.best_params_['max_depth'], xvalid_TREE, yvalid_TREE, best_model_tree, size)


# ------- Feature importance -------
importances = best_model_tree.feature_importances_
print("Feature Importance:")
for feature, importance in zip(xtrain_TREE.columns, importances):
    print(f"{feature}: {importance:.4f}")


# ------- Save the model -------
joblib.dump(best_model_tree, 'best_model_tree.pkl')


# ------- Visualizations -------
# Example: Visualizing confusion matrix
sns.heatmap(confusion_matrix(yvalid_TREE, yvalid_pred_best), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()