import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------- Initialize data and preprocessing steps -------
X_train_DF_ANN = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
Y_train_DF_ANN = Y_train_DF
scaler = StandardScaler()
size = 0.3  # Validation %

# -------Split data into training and validation sets, normalize -------
Xtrain_ANN, Xvalid_ANN, Ytrain_ANN, Yvalid_ANN = train_test_split(scaler.fit_transform(X_train_DF_ANN), Y_train_DF_ANN, test_size=size, random_state=42
)
# -------Initialize base model ------
model_ANN = MLPClassifier(random_state=42, hidden_layer_sizes=(5), max_iter=500, activation='relu', verbose=True
)

# -------Train and evaluate base model -------
model_ANN.fit(Xtrain_ANN, Ytrain_ANN)
print('Base Model Performance:')
print(f'Train Accuracy: {accuracy_score(Ytrain_ANN, model_ANN.predict(Xtrain_ANN))}')
print(f'Validation Accuracy: {accuracy_score(Yvalid_ANN, model_ANN.predict(Xvalid_ANN))}')


# ------- Grid search -------
# Perform randomized search for best model
hyperparameters_ANN = {
    'hidden_layer_sizes': [(4,), (5,), (6,), (7,), (3, 2), (4, 2), (6, 3), (7, 3), (7, 4)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate_init': [0.001, 0.0001],
    'alpha': [0.0001, 0.00001]
}

Random_search_ANN = RandomizedSearchCV(
    estimator=MLPClassifier(random_state=42, max_iter=500, verbose=True),
    param_distributions=hyperparameters_ANN,
    cv=10,
    random_state=42
)

Random_search_ANN.fit(Xtrain_ANN, Ytrain_ANN)
best_model_ANN = Random_search_ANN.best_estimator_


# ------- Train and evaluate best model -------
print('Best Model Performance:')
print(f'Train Accuracy: {accuracy_score(Ytrain_ANN, best_model_ANN.predict(Xtrain_ANN))}')
print(f'Validation Accuracy: {accuracy_score(Yvalid_ANN, best_model_ANN.predict(Xvalid_ANN))}')
print('Best Model Parameters:')
print(Random_search_ANN.best_params_)


# ------- Additional Evaluations -------
Yvalid_pred = best_model_ANN.predict(Xvalid_ANN)
print('\nClassification Report:')
print(classification_report(Yvalid_ANN, Yvalid_pred))
print('\nConfusion Matrix:')
print(confusion_matrix(Yvalid_ANN, Yvalid_pred))

# ------- Save the model -------
joblib.dump(best_model_ANN, 'best_model_ANN.pkl')


# ------- Visualizations -------
sns.heatmap(confusion_matrix(Yvalid_ANN, Yvalid_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()