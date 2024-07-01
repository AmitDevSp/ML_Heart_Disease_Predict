import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, confusion_matrix
import utility


class ANNProcessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.X_train = None
        self.Y_train = None
        self.X_test = None


    def load_and_preprocess_data(self):
        train_df = pd.read_csv(self.train_path)
        self.X_train = self.preprocess_data(train_df)
        self.Y_train = train_df[['y']]
        
        test_df = pd.read_csv(self.test_path)
        self.X_test = self.preprocess_data(test_df, is_test=True)


    def preprocess_data(self, df, is_test=False):
        df = df.drop(columns=['id'])
        df.loc[df['ca'] == 4, 'ca'] = 0
        df.loc[df['thal'] == 0, 'thal'] = 2
        df.loc[df['age'] > 120, 'age'] = 120
        df.loc[df['trestbps'] > 140, 'trestbps'] = 2
        df.loc[df['trestbps'] > 100, 'trestbps'] = 1
        df.loc[df['trestbps'] > 2, 'trestbps'] = 0

        if not is_test:
            df = df.drop(columns=['y'])
        return df


    def prepare_data_for_ann(self):
        self.X_train = pd.get_dummies(self.X_train, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
        self.X_test = pd.get_dummies(self.X_test, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
        
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)



    def train_and_predict_ann(self):
        model = MLPClassifier(hidden_layer_sizes=(7, 4), max_iter=3000, activation='relu', solver='sgd', learning_rate_init=0.001, alpha=0.00001, verbose=True)
        model.fit(self.X_train, self.Y_train.values.ravel())
        predictions = model.predict(self.X_test)
        return predictions


    def save_predictions(self, predictions, output_file):
        df_prediction = pd.DataFrame(predictions, columns=['y'])
        df_prediction.to_csv(output_file, index=False)


    def run(self):
        self.load_and_preprocess_data()
        self.prepare_data_for_ann()
        predictions = self.train_and_predict_ann()
        self.save_predictions(predictions, "Y_test_prediction.csv")


# Example usage
processor = ANNProcessor("Xy_train.csv", "X_test.csv")
processor.run()