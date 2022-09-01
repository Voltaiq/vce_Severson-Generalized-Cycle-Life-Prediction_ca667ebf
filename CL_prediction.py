from functools import reduce
import ML_models
import numpy as np
import pandas as pd
import severson_featurization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
importlib.reload(ML_models)
importlib.reload(severson_featurization)
from severson_featurization import calc_X_and_y, drop_unfinished_tests
from ML_models import TrainedModel




class CLPrediction:
    ''' 
    Class to manipulate and store information for a prediction dataset.
    This includes adding test names, storing test data, featurizing that data, 
    performing a train/test split (if relevant), etc.
    '''
    
    def __init__(self):
        self.train_list = []
        self.train_list_custom = []
        self.test_list = []
        self.test_list_custom = []
        self.predict_list = []
        self.predict_list_custom = []
        self.train_test_records = []
        self.test_test_records = []
        self.predict_test_records = []
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.X_predict = pd.DataFrame()
        self.y_predict = pd.DataFrame()
        self.avg_cyc_time = []
        self.total_cycles_predict = pd.DataFrame()
        self.capacity_retention_threshold = 0.85 # default at 0.85, might want to change this...
        self.start_cycle = 9 # also likely don't want to hard-code this in this way
        self.end_cycle = 99 # also likely don't want to hard-code this in this way
        self.last_custom_search = '' # save the last custom search for pre-population
        self.custom_search_results = []
        self.train_test_ratio = 0
        self.use_train_test_split = False
        self.ml_model = ''
        self.trained_models = {}
        self.model_predicted_cyclelife = {}
        self.model_predicted_timeleft = {}

    
    def set_train_test_list(self, train_list_add, train_vs_test):
        ''' add additional test records to the training/testing list '''
        if train_vs_test == 'train':
            self.train_list = list(train_list_add)
        elif train_vs_test == 'test':
            self.test_list = list(train_list_add)
        elif train_vs_test == 'predict':
            self.predict_list = list(train_list_add)
        
    def replace_train_test_list_custom(self, train_list_add,train_vs_test):
        ''' add additional test records to the training/testing list '''
        if train_vs_test == 'train':
            self.train_list_custom = list(train_list_add)
        elif train_vs_test == 'test':
            self.test_list_custom = list(train_list_add)
        elif train_vs_test == 'predict':
            self.predict_list_custom = list(train_list_add)
        
        
    def get_train_test_list(self, train_vs_test):
        ''' retrieve the current training/testing list'''
        if train_vs_test == 'train':
            return self.train_list
        elif train_vs_test == 'test':
            return self.test_list
        elif train_vs_test == 'predict':
            return self.predict_list
    
    def get_custom_train_test_list(self, train_vs_test):
        ''' retrieve the current custom training/testing list'''
        if train_vs_test == 'train':
            return self.train_list_custom
        elif train_vs_test == 'test':
            return self.test_list_custom
        elif train_vs_test == 'predict':
            return self.predict_list_custom

    
    def get_last_custom_search(self):
        ''' retrieve the last stored search value'''
        return self.last_custom_search
    
    def set_last_custom_search(self, search_term):
        ''' set the last search term'''
        self.last_custom_search = search_term
        
    def save_custom_search_results(self, search_results):
        ''' save a list of search results'''
        self.custom_search_results = search_results
        
    def get_custom_search_results(self):
        ''' retrieve the search results'''
        return self.custom_search_results
    
    def set_train_test_split_ratio(self, ratio):
        ''' set the train-test split ratio if sklearn's train_test_split method will be used'''
        self.train_test_ratio = ratio
        
    def get_train_test_split_ratio(self):
        ''' retrieve the current train-test-split ratio'''
        return self.train_test_ratio
    
    def set_train_test_split_option(self, option):
        ''' set whether sklearn's train_test_split method will be used'''
        self.use_train_test_split = option
    
    def get_train_test_split_option(self):
        ''' retrieve whether train_test_split will be used'''
        return self.use_train_test_split  
    
    def get_train_test_records(self):
        ''' retrieve the current training test record objects'''
        return self.train_test_records
    
    def get_test_test_records(self):
        ''' retrieve the current testing test record objects'''
        return self.test_test_records
    
    def get_predict_test_records(self):
        ''' retrieve the current testing test record objects'''
        return self.predict_test_records
    
    def populate_train_test_records(self, test_records):
        ''' function to populate training test record objects'''
        self.train_test_records = test_records
    
    def populate_test_test_records(self, test_records):
        ''' function to populate testing test record objects'''
        self.test_test_records = test_records
        
    def populate_predict_test_records(self, test_records):
        ''' function to populate testing test record objects'''
        self.predict_test_records = test_records
    
    def featurize(self,trs):
        ''' function to featurize the train and/or test data. will have to determine if train and/or test data exists'''
        # start with training data:
        self.X_train, self.y_train = calc_X_and_y(self.train_test_records,self.start_cycle, self.end_cycle, self.capacity_retention_threshold, predict = False,trs = trs)
        # either also featurize the testing dataset, or use sklearn's train test split to create the train and test featurized datasets
        if not self.use_train_test_split:
            self.X_test, self.y_test = calc_X_and_y(self.test_test_records,self.start_cycle, self.end_cycle, self.capacity_retention_threshold, predict = False,trs = trs)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=self.train_test_ratio, random_state=0)
        
        self.X_train, self.y_train = drop_unfinished_tests(self.X_train , self.y_train)
        self.X_test, self.y_test = drop_unfinished_tests(self.X_test , self.y_test)
    
    def featurize_predict(self, trs):
        ''' function to featurize the prediction data'''
#         self.populate_predict_test_records(self.predict_test_records)
        self.X_predict, self.total_cycles_predict, self.y_predict, self.avg_cyc_time = calc_X_and_y(self.predict_test_records, self.start_cycle, self.end_cycle, self.capacity_retention_threshold, predict = True,trs = trs)
    
    # likely want to set requirement that this be numerical, between 0 and 1
    def set_cap_retention_threshold(self, cap_retention):
        ''' function to set the capacity retention threshold that will be used for featurizing the dataset'''
        self.capacity_retention_threshold = cap_retention
        
    def set_start_cycle(self, cycle):
        ''' function to set the initial cycle for the featurization'''
        self.start_cycle = cycle
        
    def set_end_cycle(self, cycle):
        ''' function to set the end cycle for the featurization'''
        self.end_cycle = cycle
        
    def get_start_cycle(self):
        ''' retrieve the initial cycle for featurization'''
        return self.start_cycle
    
    def get_end_cycle(self):
        ''' retrieve the end cycle for featurization'''
        return self.end_cycle
    
    def get_featurized_data(self):
        ''' function to return X_train, X_test, y_train, y_test'''
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def set_model(self, model):
        ''' function to set the ML model to use'''
        self.ml_model = model
    
    def train_model(self):
        ''' train models listed in self.ml_model'''
        # reset the trained models dataframe
        self.trained_models = {}
        for model in self.ml_model:
            self.trained_models[model] = TrainedModel(model, (self.X_train, self.X_test, self.y_train, self.y_test))
            self.trained_models[model].train_model()

    def test_predict(self):
        ''' predict outcome for each model listed in self.ml_model'''
        for model in self.ml_model:
            self.trained_models[model].test_prediction()
    
    def predict(self):
        ''' predict outcome for each model listed in self.ml_model'''
        for model in self.ml_model:
            self.trained_models[model].predict(self.X_predict)
            
    def create_parity_plots(self):
        ''' create parity plots for all models. might want to have option for choosing model...'''
        for model in self.ml_model:
            self.trained_models[model].parity_plot()
            
    def return_model_stats(self):
        ''' return MAPE and RMSE for each model. might want to have option for choosing model...'''
        train_mape = []
        test_mape = []
        train_rmse = []
        test_rmse = []
        model_list = []
        for model in self.ml_model:
            mape = self.trained_models[model].get_MAPE()
            rmse = self.trained_models[model].get_RMSE()
            model_list.append(model)
            train_mape.append(mape[0])
            test_mape.append(mape[1])
            train_rmse.append(rmse[0])
            test_rmse.append(rmse[1])
        model_stat_df = pd.DataFrame({'Model':model_list,'Train MAPE':train_mape, 'Test MAPE': test_mape, 'Train RMSE': train_rmse, 'Test RMSE':test_rmse})
        model_stat_df.set_index('Model', inplace = True)
        return model_stat_df
    
    def plot_model_stats(self, stat_type):
        ''' function to create a bar-plot of model stats of stat_type'''
#         plt.rc('font', family='sans-serif')
#         sns.set_style("ticks")
#         sns.set_context("paper")
#         plt.rc('xtick', labelsize='medium')
#         plt.rc('ytick', labelsize='medium')
#         plt.rc('axes', titlesize='large')
#         sns.color_palette("Set2")

        model_stat_df = self.return_model_stats()
        if stat_type == 'MAPE':
            ax = model_stat_df.plot.bar(y = ['Train MAPE', 'Test MAPE'], rot = 35, ylabel = 'Mean Absolute Percentage Error (%)')#, 
                                        #figsize = (5,5), colormap = sns.color_palette("Set2", as_cmap=True),fontsize = 15)
            plt.show()
        elif stat_type == 'RMSE':
            ax = model_stat_df.plot.bar(y = ['Train RMSE', 'Test RMSE'], rot = 35, ylabel = 'Root Mean Squared Error (Cycles)')#, 
                                        #figsize = (5,5), colormap = sns.color_palette("Set2", as_cmap=True),fontsize = 10)
            plt.show()

    def calc_predicted_cyclelife(self):
        ''' function that calculates and stores predicted cycle lives in dataframes for each model; stored as a dictionary'''
        for model in self.ml_model:
            cycle_life = 10**self.trained_models[model].get_prediction(predict=True)
            names = self.X_predict.Name
            # calculate a predicted time to failure for each test:
            # (predicted cycle - current cycle) * avg cycle time
            current_cyc = self.total_cycles_predict.total_cycles
            pred_failure_time = (cycle_life - current_cyc) * self.avg_cyc_time # in hours
#             print(type(pred_failure_time))
            pred_failure_time.mask(pred_failure_time < 0, 0, inplace = True)
            self.model_predicted_cyclelife[model] = pd.DataFrame({'Name':names, 
                                                                  str(model) + ' Predicted Cycle Life':cycle_life})
            self.model_predicted_timeleft[model] = pd.DataFrame({'Name':names,
                                                                 str(model) + ' Time to Failure (h)':pred_failure_time})
        self.model_predicted_cyclelife['Combined'] = reduce(lambda x, y: pd.merge(x, y, on = 'Name'), 
                                                            [self.model_predicted_cyclelife[model] 
                                                             for model in self.ml_model])
        self.model_predicted_cyclelife['Combined']['Current cycle'] = self.total_cycles_predict.total_cycles
        self.model_predicted_cyclelife['Combined'][f'Cycle to {self.capacity_retention_threshold}% capacity retention'] = self.y_predict.cyc_life
        self.model_predicted_cyclelife['Combined'][f'Cycle to {self.capacity_retention_threshold}% capacity retention'].replace("unfinished",0,inplace=True)
        
        # combined dataframe will hold all the hour predictions
        self.model_predicted_timeleft['Combined'] = reduce(lambda x, y: pd.merge(x, y, on = 'Name'), [self.model_predicted_timeleft[model] for model in self.ml_model]) 
        idx_to_drop = self.model_predicted_cyclelife['Combined'][self.model_predicted_cyclelife['Combined'][f'Cycle to {self.capacity_retention_threshold}% capacity retention'] != 0].index
        self.model_predicted_timeleft['Combined'].drop(index = idx_to_drop, inplace = True)
        # will want to add in TIME TO FAILURE = (pred cycle - curr cycle) * avg cycle time
#         self.model_predicted_cyclelife['Combined']['Predicted time to failure (days)'] = 
        # if the number here is negative, replace with zero
        
    def return_predicted_cyclelife(self,model = 'Combined'):
        ''' function that returns a dataframe of predicted cycle life values for each test name in the prediction list'''
        return self.model_predicted_cyclelife[model], self.model_predicted_timeleft[model]

# def predict_time_to_failure(self):
#     ''' function that returns a dataframe with the test name and a predicted number of days to failure
#     If the test has already passed the capacity retention threshold then this will be zero.'''
    
#     pred_time_failure = pd.DataFrame()
#     pred_time_failure['Name'] = self.model_predicted_cyclelife['Combined']['Name']
#     current_cycles = self.total_cycles_predict.total_cycles
#     pred_failure_cycles = 
#     days_to_failure = 
    
#     pred_time_failure['Predicted time to failure (days)'] =