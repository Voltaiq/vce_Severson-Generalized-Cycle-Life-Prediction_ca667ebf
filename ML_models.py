from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from mapie.regression import MapieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
class Model:
    ''' class of trained models'''
    
    
    
    def __init__(self, model, train_test_tuple, use_mapie):
        self.model = model
        self.train_test_data = train_test_tuple
        self.pipeline = None
        self.mapie = None
        self.X_train_full = train_test_tuple[0]
        self.X_test_full = train_test_tuple[1]
        self.X_train = train_test_tuple[0]
        self.X_test = train_test_tuple[1]
        self.X_test_array = None
        self.y_train = train_test_tuple[2]
        self.y_test = train_test_tuple[3]
        self.X_predict = ()
        self.X_test_array = None
        self.train_predict = None
        self.train_pis = None
        self.test_predict = None
        self.test_pis = None
        self.predict_predict = None
        self.predict_pis = None
        self.l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        self.mapie_error = use_mapie
        
    def get_train_test_data(self):
        return self.train_test_data
    
    def get_model(self):
        return self.model
    
    def train_model(self):
        ''' function to train the model. Training will depend on model type'''
        # add use mapie logic here
        
        if self.model == 'Severson variance':
            print("Training Severson variance model")
            self.pipeline = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(l1_ratio=self.l1_ratios, cv=5, random_state=0, max_iter = 25000))])

            self.X_train = self.X_train['var_deltaQ']
            self.X_test = self.X_test['var_deltaQ']
            self.X_test_array = np.array(self.X_test).reshape(-1, 1)
            if self.mapie_error:
                self.mapie = MapieRegressor(self.pipeline, method="plus", cv=10)

                self.mapie.fit(np.array(self.X_train).reshape(-1, 1), np.ravel(self.y_train))
                self.train_predict, self.train_pis = self.mapie.predict(np.array(self.X_train).reshape(-1, 1), alpha=[0.05])
            else:
                self.pipeline.fit(np.array(self.X_train).reshape(-1, 1), np.ravel(self.y_train))
                self.train_predict = self.pipeline.predict(np.array(self.X_train).reshape(-1, 1))
                

            print("Completed training Severson variance model")
#             self.test_predict = self.pipeline.predict(np.array(self.X_test).reshape(-1, 1))

        elif self.model == 'Severson discharge':
            print("Training Severson discharge model")
            self.X_train = self.X_train.drop(columns = ['Name','Dataset_group']) #'deltaQ_lowV'
            self.X_test = self.X_test.drop(columns = ['Name','Dataset_group']) #,'deltaQ_lowV'
            self.X_test_array = np.array(self.X_test)

            self.pipeline =  Pipeline([('scaler', StandardScaler()), ('enet', ElasticNetCV(l1_ratio=self.l1_ratios, cv=5, random_state=0, max_iter = 25000))])
            if self.mapie_error:
                self.mapie = MapieRegressor(self.pipeline, method="plus", cv=10)
                # also train the pipeline itself to be able to access the named steps
                self.pipeline.fit(np.array(self.X_train), np.ravel(self.y_train))
                self.mapie.fit(np.array(self.X_train), np.ravel(self.y_train))
                

                self.train_predict, self.train_pis = self.mapie.predict(np.array(self.X_train), alpha=[0.05])
            else:
                self.pipeline.fit(np.array(self.X_train), np.ravel(self.y_train))
                self.train_predict = self.pipeline.predict(np.array(self.X_train))
            print("Completed training Severson discharge model")
#             self.test_predict = self.pipeline.predict(np.array(self.X_test))

        elif self.model == 'Dummy':
            print("Training Dummy model")
            self.pipeline = Pipeline([('dummy',DummyRegressor())])
            self.X_test_array = self.X_test
            if self.mapie_error:
                self.mapie = MapieRegressor(self.pipeline, method="plus", cv=10)
                self.mapie.fit(self.X_train, self.y_train)

                self.train_predict, self.train_pis = self.mapie.predict(self.X_train, alpha=[0.05])
            else:
                self.pipeline.fit(self.X_train, self.y_train)
                self.train_predict = self.pipeline.predict(self.X_train)
#             self.test_predict = self.dummy_regr.predict(self.X_test)
            print("Completed training Dummy model")
    
        elif self.model == "Severson discharge XGBoost":
            print("Training XGBoost model on Severson discharge features")
            self.X_train = self.X_train.drop(columns = ['Name','Dataset_group']) #'deltaQ_lowV'
            self.X_test = self.X_test.drop(columns = ['Name','Dataset_group']) #,'deltaQ_lowV'
            self.X_test_array = np.array(self.X_test)

            # would likely be great to include some sort of gridsearch of tunable parameters...
            self.pipeline =  Pipeline([('scaler', StandardScaler()), ('xgboost', xgb.XGBRegressor(max_depth=10,n_estimators=50))])
            if self.mapie_error:
                self.mapie = MapieRegressor(self.pipeline, method="plus", cv=10)
            
                self.mapie.fit(np.array(self.X_train), np.ravel(self.y_train))

                self.train_predict, self.train_pis = self.mapie.predict(np.array(self.X_train), alpha=[0.05])
            else:
                self.pipeline.fit(np.array(self.X_train), np.ravel(self.y_train))
                self.train_predict = self.pipeline.predict(np.array(self.X_train))
            print("Completed training XGBoost model on Severson discharge features")
    
    def test_prediction(self):
        ''' function to predict outputs of test data'''
        if self.mapie_error:
            self.test_predict, self.test_pis = self.mapie.predict(self.X_test_array, alpha=[0.05])
        else:
            self.test_predict = self.pipeline.predict(self.X_test_array)
        print("Completed predicting test output for " + self.model + " model")

    def predict(self, X_predict):
        ''' function to predict outputs of prediction data'''
        self.X_predict = X_predict
        if self.model == 'Severson variance':
            self.X_predict = self.X_predict['var_deltaQ']
            self.X_predict_array = np.array(self.X_predict).reshape(-1, 1)
        elif self.model == 'Severson discharge':
            self.X_predict = self.X_predict.drop(columns = ['Name','Dataset_group'])
            self.X_predict_array = np.array(self.X_predict)
        elif self.model == 'Dummy':
            self.X_predict_array = self.X_predict
        elif self.model == "Severson discharge XGBoost":
            self.X_predict = self.X_predict.drop(columns = ['Name','Dataset_group'])
            self.X_predict_array = np.array(self.X_predict)
        if self.mapie_error:
            self.predict_predict, self.predict_pis = self.mapie.predict(self.X_predict_array, alpha=[0.05])
        else:
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
    
    def get_grouped_MAPE(self,train_vs_test, idx = None):
        ''' return mean absolute percentage error. Returns a tuple of (train, test) errors.'''
        if train_vs_test == 'train':
            return 100*mean_absolute_percentage_error(np.power(10, self.y_train), np.power(10, self.train_predict))
        else:
            return 100*mean_absolute_percentage_error(np.power(10, self.y_test.log_cyc_life[idx]), np.power(10, self.test_predict[idx]))
        
    
    def get_grouped_RMSE(self):
        ''' return root mean squared error. Returns a tuple of (train, test) errors.'''
        train_rmse = mean_squared_error(np.power(10, self.y_train), np.power(10, self.train_predict), squared=False)
        test_rmse = mean_squared_error(np.power(10, self.y_test), np.power(10, self.test_predict), squared=False)
        return (train_rmse, test_rmse)

    
    def parity_plot(self):
        ''' create a parity plot of the real vs predicted values. Will plot train and test predictions'''
        
        unique_groups = pd.unique(self.X_test_full.Dataset_group)
        train_unique_groups = pd.unique(self.X_train_full.Dataset_group)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for group in train_unique_groups:
            train_idx = self.X_train_full[self.X_train_full.Dataset_group == group].index
            model_resetidx = self.X_train_full.reset_index()
            train_reset_idx = model_resetidx[model_resetidx['index'].isin(train_idx)].index
            if self.mapie_error:
                lower_bounds = self.train_pis[:, 0, 0].T[train_reset_idx]
                upper_bounds = self.train_pis[:, 1, 0].T[train_reset_idx]
                y_err = np.abs([10**lower_bounds, 10**upper_bounds] - 10**self.train_predict[train_reset_idx])
                plt.errorbar(x = 10**self.y_train.log_cyc_life[train_idx],
                             y = 10**self.train_predict[train_reset_idx],
                             yerr = y_err, capsize = 3,linestyle='',
                            label = 'Train '+group, marker = 'o',alpha = 0.6)#, c = '#68CCCA')
            else:
                plt.scatter(x = 10**self.y_train.log_cyc_life[train_idx],
                             y = 10**self.train_predict[train_reset_idx],
                            label = 'Train '+group, marker = 'o',alpha = 0.6)
        for group in unique_groups:            
            index = self.X_test_full[self.X_test_full.Dataset_group == group].index
            model_resetidx = self.X_test_full.reset_index()
            reset_idx = model_resetidx[model_resetidx['index'].isin(index)].index
            if self.mapie_error:
                lower_bounds = self.test_pis[:, 0, 0].T[reset_idx]
                upper_bounds = self.test_pis[:, 1, 0].T[reset_idx]
                y_err = np.abs([10**lower_bounds, 10**upper_bounds] - 10**self.test_predict[reset_idx])
                plt.errorbar(x = 10**self.y_test.log_cyc_life[index],
                            y = 10**self.test_predict[reset_idx],
                            yerr = y_err, capsize = 3,linestyle='',
                            label = 'Test '+group, alpha = 0.6,marker='v')#,c = '#FDA1FF')
            else:
                plt.scatter(x = 10**self.y_test.log_cyc_life[index],
                            y = 10**self.test_predict[reset_idx],
                            label = 'Test '+group, alpha = 0.6,marker='v')
        max_axis = 10**max([max(self.y_train.log_cyc_life),
                                max(self.y_test.log_cyc_life),
                                max(self.train_predict),
                                max(self.test_predict)])
        min_axis = 10**min([min(self.y_train.log_cyc_life),
                                min(self.y_test.log_cyc_life),
                                min(self.train_predict),
                                min(self.test_predict)])
        plt.plot([min_axis,max_axis],[min_axis,max_axis],'k')
        plt.legend(bbox_to_anchor=(1, 0.5),loc = 'center left') # uncomment to get legend
        # ax.set_ylim(0,5000)
        # ax.set_xlim(0,5000)
        plt.xlabel('Observed cycle life')
        plt.ylabel('Predicted cycle life')
        plt.title("Parity plot of " + self.model + " model")
        plt.axis('square')
        plt.show()