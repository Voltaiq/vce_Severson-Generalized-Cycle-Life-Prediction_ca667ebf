from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.graph_objects as go

    
class TrainedModel:
    ''' class of trained models'''
    
    
    
    def __init__(self, model, train_test_tuple):
        self.model = model
        self.train_test_data = train_test_tuple
        self.pipeline = None
        self.X_train = train_test_tuple[0]
        self.X_test = train_test_tuple[1]
        self.X_test_array = None
        self.y_train = train_test_tuple[2]
        self.y_test = train_test_tuple[3]
        self.X_predict = ()
        self.X_test_array = None
        self.train_predict = None
        self.test_predict = None
        self.predict_predict = None
        self.l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        
    def get_train_test_data(self):
        return self.train_test_data
    
    def get_model(self):
        return self.model
    
    @ignore_warnings(category=ConvergenceWarning)
    def train_model(self):
        ''' function to train the model. Training will depend on model type'''
        if self.model == 'Severson variance':
            print("Training Severson variance model")
            self.pipeline = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(l1_ratio=self.l1_ratios, cv=5, random_state=0))])

            self.X_train = self.X_train['var_deltaQ']
            self.X_test = self.X_test['var_deltaQ']
            self.X_test_array = np.array(self.X_test).reshape(-1, 1)

            self.pipeline.fit(np.array(self.X_train).reshape(-1, 1), np.ravel(self.y_train))
            self.train_predict = self.pipeline.predict(np.array(self.X_train).reshape(-1, 1))


            print("Completed training Severson variance model")
#             self.test_predict = self.pipeline.predict(np.array(self.X_test).reshape(-1, 1))

        elif self.model == 'Severson discharge':
            print("Training Severson discharge model")
            self.X_train = self.X_train.drop(columns = ['Name','deltaQ_lowV'])
            self.X_test = self.X_test.drop(columns = ['Name','deltaQ_lowV'])
            self.X_test_array = np.array(self.X_test)

            self.pipeline =  Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(l1_ratio=self.l1_ratios, cv=5, random_state=0))])
            self.pipeline.fit(np.array(self.X_train), np.ravel(self.y_train))

            self.train_predict = self.pipeline.predict(np.array(self.X_train))
            print("Completed training Severson discharge model")
#             self.test_predict = self.pipeline.predict(np.array(self.X_test))

        elif self.model == 'Dummy':
            print("Training Dummy model")
            self.pipeline = Pipeline([('dummy',DummyRegressor())])
            self.pipeline.fit(self.X_train, self.y_train)
            self.X_test_array = self.X_test

            self.train_predict = self.pipeline.predict(self.X_train)
#             self.test_predict = self.dummy_regr.predict(self.X_test)
            print("Completed training Dummy model")
    
    def test_prediction(self):
        ''' function to predict outputs of test data'''
        self.test_predict = self.pipeline.predict(self.X_test_array)
        print("Completed predicting test output for " + self.model + " model")

    def predict(self, X_predict):
        ''' function to predict outputs of prediction data'''
        self.X_predict = X_predict
        if self.model == 'Severson variance':
            self.X_predict = self.X_predict['var_deltaQ']
            self.X_predict_array = np.array(self.X_predict).reshape(-1, 1)
        elif self.model == 'Severson discharge':
            self.X_predict = self.X_predict.drop(columns = ['Name','deltaQ_lowV'])
            self.X_predict_array = np.array(self.X_predict)
        elif self.model == 'Dummy':
            self.X_predict_array = self.X_predict
        self.predict_predict = self.pipeline.predict(self.X_predict_array)
        print("Completed predicting outcome for " + self.model + " model")
    
    def get_prediction(self, predict = False):
        ''' function to retrieve prediction data from model. Returns a tuple of (train_predict, test_predict)'''
        if not predict:
            return (self.train_predict, self.test_predict)
        else:
            return self.predict_predict
    
    def get_MAPE(self):
        ''' return mean absolute percentage error. Returns a tuple of (train, test) errors.'''
        train_mape = 100*mean_absolute_percentage_error(np.power(10, self.y_train), np.power(10, self.train_predict))
        test_mape = 100*mean_absolute_percentage_error(np.power(10, self.y_test), np.power(10, self.test_predict))
        return (train_mape, test_mape)
    
    def get_RMSE(self):
        ''' return root mean squared error. Returns a tuple of (train, test) errors.'''
        train_rmse = mean_squared_error(np.power(10, self.y_train), np.power(10, self.train_predict), squared=False)
        test_rmse = mean_squared_error(np.power(10, self.y_test), np.power(10, self.test_predict), squared=False)
        return (train_rmse, test_rmse)

    
    def parity_plot(self):
        ''' create a parity plot of the real vs predicted values. Will plot train and test predictions'''
#         plt.rc('font', family='serif')
#         plt.rc('xtick', labelsize='x-small')
#         plt.rc('ytick', labelsize='x-small')
#         sns.set_style("ticks")
#         sns.set_context("paper")
#         sns.color_palette("Set2")
#         fig = go.Figure()
#         fig.add_traces(go.Scatter(x = 10**self.y_train,y = 10**self.train_predict, name = 'Train', opacity = 0.6))
#         fig.show()
                      
        
        plt.scatter(10**self.y_train,10**self.train_predict, label = 'Train', alpha = 0.6)#, c = '#68CCCA')
        plt.scatter(10**self.y_test,10**self.test_predict, label = 'Test', alpha = 0.6)#,c = '#FDA1FF')
        max_axis = 10**max([max(self.y_train.log_cyc_life),max(self.y_test.log_cyc_life),max(self.train_predict),max(self.test_predict)])
        plt.plot([0,max_axis],[0,max_axis])
        plt.legend(bbox_to_anchor=(1, 0.5),loc = 'center left')
        plt.xlabel('Observed cycle life')
        plt.ylabel('Predicted cycle life')
        plt.title("Parity plot on " + self.model + " model")
        plt.axis('square')
        plt.show()
        
