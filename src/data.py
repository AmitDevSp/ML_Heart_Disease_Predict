import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.discretisers import EqualWidthDiscretiser


class DataProcessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.trainDF = None
        self.testDF = None


    def load_data(self):
        try:
            self.trainDF = pd.read_csv(self.train_path)
            self.testDF = pd.read_csv(self.test_path)
        except Exception as e:
            print(f"Error loading data: {e}")


    def preprocess_data(self):
        self.trainDF = self.trainDF.drop(columns=['id'])
        self.testDF = self.testDF.drop(columns=['id'])
        
        self.trainDF.loc[self.trainDF['ca'] == 4, 'ca'] = 0
        self.trainDF.loc[self.trainDF['thal'] == 0, 'thal'] = 2
        self.trainDF.loc[self.trainDF['age'] > 120, 'age'] = 120
        self.trainDF.loc[self.trainDF['trestbps'] > 140, 'trestbps'] = 2
        self.trainDF.loc[self.trainDF['trestbps'] > 100, 'trestbps'] = 1
        self.trainDF.loc[self.trainDF['trestbps'] > 2, 'trestbps'] = 0

        self.testDF.loc[self.testDF['ca'] == 4, 'ca'] = 0
        self.testDF.loc[self.testDF['thal'] == 0, 'thal'] = 2
        self.testDF.loc[self.testDF['age'] > 120, 'age'] = 120
        self.testDF.loc[self.testDF['trestbps'] > 140, 'trestbps'] = 2
        self.testDF.loc[self.testDF['trestbps'] > 100, 'trestbps'] = 1
        self.testDF.loc[self.testDF['trestbps'] > 2, 'trestbps'] = 0


    def count_age_groups(self):
        age_groups = [0, 0, 0, 0, 0, 0, 0]
        for i in self.trainDF['age']:
            if i <= 30:
                age_groups[0] += 1
            elif i <= 40:
                age_groups[1] += 1
            elif i <= 50:
                age_groups[2] += 1
            elif i <= 60:
                age_groups[3] += 1
            elif i <= 70:
                age_groups[4] += 1
            elif i <= 80:
                age_groups[5] += 1
            else:
                age_groups[6] += 1
        return age_groups


    def count_cp_types(self):
        cp_counts = [0, 0, 0, 0]
        for i in self.trainDF['cp']:
            if i == 0:
                cp_counts[0] += 1
            elif i == 1:
                cp_counts[1] += 1
            elif i == 2:
                cp_counts[2] += 1
            elif i == 3:
                cp_counts[3] += 1
        return cp_counts


    def count_gender(self):
        count_male = (self.trainDF['gender'] == 1).sum()
        count_female = (self.trainDF['gender'] == 0).sum()
        return count_female, count_male


    def count_trestbps(self):
        low_blood_pressure = (self.trainDF['trestbps'] <= 100).sum()
        proper_blood_pressure = ((self.trainDF['trestbps'] > 100) & (self.trainDF['trestbps'] <= 140)).sum()
        high_blood_pressure = (self.trainDF['trestbps'] > 140).sum()
        return low_blood_pressure, proper_blood_pressure, high_blood_pressure


    def plot_trestbps(self, counts):
        plt.bar(x=["Low BP", "Proper BP", "High BP"], height=counts)
        plt.xlabel('Trestbps')
        plt.ylabel('Amount')
        plt.show()


    def plot_mosaic(self):
        mosaic(self.trainDF, ['fbs', 'thal'])
        plt.show()
        mosaic(self.trainDF, ['fbs', 'ca'])
        plt.show()
        mosaic(self.trainDF, ['restecg', 'slope'])
        plt.show()
        mosaic(self.trainDF, ['cp', 'restecg'])
        plt.show()


    def plot_scatter(self):
        plt.scatter(self.trainDF['trestbps'], self.trainDF['chol'])
        plt.scatter(self.trainDF['trestbps'], self.trainDF['thalach'])
        plt.scatter(self.trainDF['trestbps'], self.trainDF['oldpeak'])
        plt.scatter(self.trainDF['chol'], self.trainDF['trestbps'])
        plt.scatter(self.trainDF['chol'], self.trainDF['thalach'])
        plt.scatter(self.trainDF['chol'], self.trainDF['oldpeak'])
        plt.scatter(self.trainDF['thalach'], self.trainDF['trestbps'])
        plt.scatter(self.trainDF['thalach'], self.trainDF['chol'])
        plt.scatter(self.trainDF['thalach'], self.trainDF['oldpeak'])
        plt.scatter(self.trainDF['oldpeak'], self.trainDF['trestbps'])
        plt.scatter(self.trainDF['oldpeak'], self.trainDF['chol'])
        plt.scatter(self.trainDF['oldpeak'], self.trainDF['thalach'])
        plt.show()


    def plot_heatmap(self):
        sns.heatmap(self.trainDF.drop(columns=['y']).corr(), annot=True, cmap='coolwarm')
        plt.show()


    def plot_distribution(self):
        for column in ['chol', 'trestbps', 'thalach', 'oldpeak']:
            sns.distplot(self.trainDF[column], color="skyblue")
            plt.show()


    def plot_kde(self):
        for column in ['chol', 'trestbps', 'thalach', 'oldpeak']:
            sns.kdeplot(self.trainDF[column], shade=True, bw=.5, color="olive")
            plt.show()


    def plot_histogram(self):
        self.trainDF['oldpeak'].hist(rwidth=0.9, color='#607c8e')
        plt.xlabel('Oldpeak')
        plt.ylabel('Amount')
        plt.grid(axis='x', alpha=0.75)
        plt.show()


    def plot_cp(self, counts):
        plt.bar(x=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], height=counts)
        plt.show()


    def plot_gender(self, counts):
        plt.bar(x=["Female", "Male"], height=counts)
        plt.show()


    def plot_age_groups(self, counts):
        plt.bar(x=["0-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"], height=counts)
        plt.xlabel('Age')
        plt.ylabel('Amount')
        plt.show()


    def plot_age_vs_trestbps(self):
        plt.scatter(self.trainDF['age'], self.trainDF['trestbps'])
        plt.show()


    def plot_age_vs_chol(self):
        plt.scatter(self.trainDF['age'], self.trainDF['chol'])
        plt.show()


    def plot_age_boxplot(self):
        self.trainDF.loc[self.trainDF['age'] > 80, 'age'] = 80
        sns.boxplot(x='y', y='age', data=self.trainDF)
        plt.show()


    def minimize_table(self):
        tempDF = self.trainDF.drop(columns=['id', 'age', 'gender', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        return tempDF



processor = DataProcessor("Xy_train.csv", "X_test.csv")
processor.load_data()
processor.preprocess_data()

# ------- Counting and plotting examples -------s
age_counts = processor.count_age_groups()
cp_counts = processor.count_cp_types()
gender_counts = processor.count_gender()
trestbps_counts = processor.count_trestbps()

processor.plot_trestbps(trestbps_counts)
processor.plot_mosaic()
processor.plot_scatter()
processor.plot_heatmap()
processor.plot_distribution()
processor.plot_kde()
processor.plot_histogram()
processor.plot_cp(cp_counts)
processor.plot_gender(gender_counts)
processor.plot_age_groups(age_counts)
processor.plot_age_vs_trestbps()
processor.plot_age_vs_chol()
processor.plot_age_boxplot()

minimized_table = processor.minimize_table()
print(minimized_table.head())
