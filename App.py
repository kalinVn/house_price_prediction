import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
# import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')


import numpy as np


class App:

    # initiating ( learning rate and number of iterations )
    def __init__(self):
        self.house_price_dataframe = None
        self.x = None
        self.y = None
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load(self):
        house_price_dataset = sklearn.datasets.load_boston()
        self.house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
        self.house_price_dataframe['price'] = house_price_dataset.target

    def splitting_data(self):
        self.x = self.house_price_dataframe.drop(['price'], axis=1)
        self.y = self.house_price_dataframe['price']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=2)

    def fit(self):
        self.model = XGBRegressor()
        # Model Training
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        training_data_prediction = self.model.predict(self.x_train)

        score_1 = metrics.r2_score(self.y_train, training_data_prediction)

        score_2 = metrics.mean_absolute_error(self.y_train, training_data_prediction)

        print("R squared error: ", score_1)
        print("Mean absolute error: ", score_2)

        test_data_prediction = self.model.predict(self.x_test)

        score_1 = metrics.r2_score(self.y_test, test_data_prediction)

        score_2 = metrics.mean_absolute_error(self.y_test, test_data_prediction)

        print("R squared error: ", score_1)
        print("Mean absolute error: ", score_2)

        self.visualize_actual_predicted_price( training_data_prediction)

    def  visualize_actual_predicted_price(self,training_data_prediction):
        plt.scatter(self.y_train, training_data_prediction)
        plt.xlabel('Actual price')
        plt.ylabel('Predicted price')
        plt.title('Actual price vs Predicted Price')
        plt.show()

