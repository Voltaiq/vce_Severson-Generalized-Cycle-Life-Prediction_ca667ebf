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
    Class to manipulate and store information for a prediction session.
    This includes adding test names, storing test data, featurizing that data, 
    performing a train/test split (if relevant), etc.
    '''
    
    def __init__(self):
        self.train_list = []
        self.train_list_custom = []
        self.test_list = []
        self.test_list_custom = []
        self.train_dataset_list = []
        self.test_dataset_list = []
        self.predict_list = []
        self.predict_list_custom = []
        self.predict_dataset_list = []
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
        self.capacity_retention_threshold = 85 # default at 0.85, might want to change this...
        self.start_cycle = 9 # also likely don't want to hard-code this in this way
        self.end_cycle = 99 # also likely don't want to hard-code this in this way
        self.reference_cycle = 1
        self.last_custom_search = '' # save the last custom search for pre-population
        self.custom_search_results = []
        self.train_test_ratio = 0
        self.use_train_test_split = False
        self.ml_model = ''
        self.trained_models = {}
        self.model_predicted_cyclelife = {}
        self.model_predicted_cyclelife_errors = {}
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
            self.train_list_custom.extend(list(train_list_add))
            train_set = set(self.train_list_custom)
            self.train_list_custom = list(train_set)
        elif train_vs_test == 'test':
            self.test_list_custom.extend(list(train_list_add))
            test_set = set(self.test_list_custom)
            self.test_list_custom = list(test_set)
        elif train_vs_test == 'predict':
            self.predict_list_custom.extend(list(train_list_add))
            predict_set = set(self.predict_list_custom)
            self.predict_list_custom = list(predict_set)
    
    def remove_train_test_list_custom(self, list_remove, train_vs_test):
        for name in list_remove:
            try:
                if train_vs_test == 'train':
                    self.train_list_custom.remove(name)
                elif train_vs_test == 'test':
                    self.test_list_custom.remove(name)
                elif train_vs_test == 'predict':
                    self.predict_list_custom.remove(name)
            except:
                print("Test " + name + " does not exist in " + train_vs_test + " dataset and can't be removed. Skipping.")
        
    def clear_train_test_list_custom(self,train_vs_test):
        if train_vs_test == 'train':
            self.train_list_custom = []
        elif train_vs_test == 'test':
            self.test_list_custom = []
        elif train_vs_test == 'predict':
            self.predict_list_custom = []
        
        
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
    
    def populate_train_test_records(self, test_records,dataset_list):
        ''' function to populate training test record objects'''
        self.train_test_records = test_records
        self.train_dataset_list = dataset_list
    
    def populate_test_test_records(self, test_records,dataset_list):
        ''' function to populate testing test record objects'''
        self.test_test_records = test_records
        self.test_dataset_list = dataset_list
        
    def populate_predict_test_records(self, test_records,dataset_list):
        ''' function to populate testing test record objects'''
        self.predict_test_records = test_records
        self.predict_dataset_list = dataset_list
        
    def set_reference_cycle(self, ref_cyc):
        ''' set reference cycle for featurization'''
        self.reference_cycle = int(ref_cyc)
    
    def featurize(self,trs):
        ''' function to featurize the train and/or test data. will have to determine if train and/or test data exists'''
        # start with training data:
        self.X_train, self.y_train = calc_X_and_y(self.train_test_records,self.start_cycle, self.end_cycle, self.capacity_retention_threshold, predict = False,trs = trs,dataset_group = self.train_dataset_list,ref_cyc = self.reference_cycle)
        # either also featurize the testing dataset, or use sklearn's train test split to create the train and test featurized datasets
        if not self.use_train_test_split:
            self.X_test, self.y_test = calc_X_and_y(self.test_test_records,self.start_cycle, self.end_cycle, self.capacity_retention_threshold, predict = False,trs = trs,dataset_group = self.test_dataset_list,ref_cyc = self.reference_cycle)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train,
                                                                                    self.y_train,                                                                                   test_size=self.train_test_ratio,
                                                                                        random_state=1)

        self.X_train, self.y_train = drop_unfinished_tests(self.X_train , self.y_train)
        self.X_test, self.y_test = drop_unfinished_tests(self.X_test , self.y_test)
    
    def featurize_predict(self, trs):
        ''' function to featurize the prediction data'''
#         self.populate_predict_test_records(self.predict_test_records)
        self.X_predict, self.total_cycles_predict, self.y_predict, self.avg_cyc_time = calc_X_and_y(self.predict_test_records, self.start_cycle, self.end_cycle, self.capacity_retention_threshold, predict = True,trs = trs,dataset_group = self.predict_dataset_list,ref_cyc = self.reference_cycle)
    
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
            self.trained_models[model] = TrainedModel(model, (self.X_train, self.X_test, 
                                                              self.y_train, self.y_test))
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
        train_mape_stdev = []
        test_mape_stdev = []
        train_rmse_stdev = []
        test_rmse_stdev = []
        for model in self.ml_model:
            mape = self.trained_models[model].get_MAPE()
            rmse = self.trained_models[model].get_RMSE()
            model_list.append(model)
            train_mape.append(mape[0])
            test_mape.append(mape[1])
            train_rmse.append(rmse[0])
            test_rmse.append(rmse[1])
            train_mape_stdev.append(0)
            test_mape_stdev.append(0)
            train_rmse_stdev.append(0)
            test_rmse_stdev.append(0)
        
        model_stat_df = pd.DataFrame({'Model':model_list,'Train MAPE':train_mape, 'Test MAPE': test_mape, 'Train RMSE': train_rmse, 'Test RMSE':test_rmse,'Train MAPE stdev':train_mape_stdev, 'Test MAPE stdev':test_mape_stdev, 'Train RMSE stdev':train_rmse_stdev, 'Test RMSE stdev':test_rmse_stdev})
        model_stat_df.set_index('Model', inplace = True)
        return model_stat_df
    
    def return_grouped_model_stats(self):
        ''' return MAPE and RMSE for each model. group based on groups in test dataset'''
        train_mape = []
        test_mape = {}
        train_rmse = []
        test_rmse = []
        model_list = []
        first=True
        unique_groups = pd.unique(self.X_test.Dataset_group)

        for model in self.ml_model:
            train_mape.append(self.trained_models[model].get_grouped_MAPE("train"))
            for group in unique_groups:
                idx = self.X_test[[g in group for g in self.X_test.Dataset_group]].index
                if first:
                    test_mape['Test MAPE ' + group]=[]
                test_mape['Test MAPE ' + group].append(self.trained_models[model].get_grouped_MAPE("test",idx))
            first=False
            rmse = self.trained_models[model].get_RMSE()
            model_list.append(model)
#             train_mape.append(mape[0])
#             test_mape.append(mape[1])
            train_rmse.append(rmse[0])
            test_rmse.append(rmse[1])
        df_dict = {'Model':model_list,'Train MAPE':train_mape,'Train RMSE': train_rmse, 'Test RMSE':test_rmse}
        df_dict.update(test_mape)
        model_stat_df = pd.DataFrame(df_dict)
        model_stat_df.set_index('Model', inplace = True)
        return model_stat_df
    
    def plot_model_stats(self, stat_type):
        ''' function to create a bar-plot of model stats of stat_type'''

        model_stat_df = self.return_model_stats()
        if stat_type == 'MAPE':
            ax = model_stat_df.plot.bar(y = ['Train MAPE', 'Test MAPE'],rot = 35, ylabel = 'Mean Absolute Percentage Error (%)')
            for container in ax.containers:
                ax.bar_label(container,labels=container.datavalues.round(1), padding=3,rotation=90)
            plt.show()
        elif stat_type == 'RMSE':
            ax = model_stat_df.plot.bar(y = ['Train RMSE', 'Test RMSE'], rot = 35, ylabel = 'Root Mean Squared Error (Cycles)')
            plt.show()

    def plot_grouped_model_stats(self, stat_type):
        model_stat_df = self.return_grouped_model_stats()
        if stat_type == 'MAPE':
            swapped_df = model_stat_df.transpose().drop(['Train RMSE','Test RMSE'])
            ax = swapped_df.plot.bar(rot = 90, ylabel = 'Mean Absolute Percentage Error (%)')
            for container in ax.containers:
                ax.bar_label(container,labels=container.datavalues.round(1), padding=3,rotation=90)
            ymin, ymax = ax.get_ylim()
            if ymax > 100:
                ax.set_yscale('log')
            else:
                ax.set_ylim(ymin, ymax + (ymax-ymin)*.05)
            plt.show()
        elif stat_type == 'RMSE':
            ax = model_stat_df.plot.bar(y = ['Train RMSE', 'Test RMSE'], rot = 35, 
                                        ylabel = 'Root Mean Squared Error (Cycles)')
            plt.show()
    
    def calc_predicted_cyclelife(self):
        ''' function that calculates and stores predicted cycle lives in dataframes for each model; stored as a dictionary'''
        for model in self.ml_model:
            cycle_life = 10**self.trained_models[model].get_prediction(predict=True)
            upper_bounds = 10**self.trained_models[model].predict_pis[:, 1, 0].T
            lower_bounds = 10**self.trained_models[model].predict_pis[:, 0, 0].T
            lower_err = np.abs(lower_bounds - cycle_life)
            upper_err = np.abs(upper_bounds - cycle_life)
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
            self.model_predicted_cyclelife_errors[model] = [lower_err, upper_err]
            
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
        
        self.model_predicted_cyclelife_errors['Combined'] = [self.model_predicted_cyclelife_errors[model] for model in self.ml_model]
        # will want to add in TIME TO FAILURE = (pred cycle - curr cycle) * avg cycle time
#         self.model_predicted_cyclelife['Combined']['Predicted time to failure (days)'] = 
        # if the number here is negative, replace with zero
        
    def return_predicted_cyclelife(self,model = 'Combined'):
        ''' function that returns a dataframe of predicted cycle life values, and array of error values, and a dataframe indicating the amount of time predicted for each test name in the prediction list'''
        return self.model_predicted_cyclelife[model], self.model_predicted_cyclelife_errors[model], self.model_predicted_timeleft[model]
    
    def get_predicted_cyclelife_with_error(self):
        ''' function that returns a dataframe of predicted cycle life values including error bar information'''
        pred_cycle_life = self.model_predicted_cyclelife['Combined'].copy()
        for i, model in enumerate(self.ml_model):
            lower_error = self.model_predicted_cyclelife_errors['Combined'][i][0]
            upper_error = self.model_predicted_cyclelife_errors['Combined'][i][1]
            pred_cycle_life[model+' Predicted Cycle Life error'] = [[lower_error[j], upper_error[j]] for j in range(len(lower_error))]
        pred_cycle_life = pred_cycle_life.reindex(sorted(pred_cycle_life.columns), axis=1)
        pred_cycle_life = pred_cycle_life.set_index('Name')
        return pred_cycle_life
        

    def grouped_feature_distribution(self, feature):
        ''' function that returns a grouped box plot of the feature distribution for a specific chosen feature'''
        train_df = self.X_train[['Dataset_group',feature]].copy()
        test_df = self.X_test[['Dataset_group',feature]].copy()
        train_df.Dataset_group = 'Train ' + train_df.Dataset_group
        test_df.Dataset_group = 'Test ' + test_df.Dataset_group
        pred_concat = pd.concat([train_df,test_df])
        ax = sns.boxplot(x = feature,y='Dataset_group', data=pred_concat,palette="Set2")
        ax = sns.stripplot(x = feature,y='Dataset_group',data=pred_concat, color=".3",alpha=0.7)
        
        # annotate with the median value for each boxplot
        group_list = pd.unique(pred_concat.Dataset_group)
        medians = pred_concat.groupby('Dataset_group')[feature].median()
        
        for i,ytick in enumerate(ax.get_yticks()):
            x_ax_range = ax.get_xlim()[1]-ax.get_xlim()[0]
            x_location = ax.get_xlim()[1] + .05*x_ax_range
            plt.text(x_location, ytick, str(round(medians[group_list[i]],1)), ha='center')

        plt.show()
        
    def return_prediction_dataframes(self, train_test):
        ''' return prediciton dataframes including test name, actual cycle life, predicted cycle life for a model'''
        pred_dict = {}
        if train_test == 'train':
            for model in self.ml_model:
                pred_dict[model] = pd.DataFrame()
                pred_dict[model]['Name'] = self.X_train['Name']
                cycle_life = np.power(10,self.trained_models[model].get_prediction()[0])
                upper_bounds = 10**self.trained_models[model].train_pis[:, 1, 0].T
                lower_bounds = 10**self.trained_models[model].train_pis[:, 0, 0].T
                lower_err = np.abs(lower_bounds - cycle_life)
                upper_err = np.abs(upper_bounds - cycle_life)
                pred_dict[model][model+' Predicted cycle life'] = cycle_life
                pred_dict[model][model+' Predicted CL error'] = [[lower_err[i], upper_err[i]] for i in range(len(lower_err))]
        elif train_test == 'test':
            for model in self.ml_model:
                pred_dict[model] = pd.DataFrame()
                pred_dict[model]['Name'] = self.X_test['Name']
                cycle_life = np.power(10,self.trained_models[model].get_prediction()[1])
                upper_bounds = 10**self.trained_models[model].test_pis[:, 1, 0].T
                lower_bounds = 10**self.trained_models[model].test_pis[:, 0, 0].T
                lower_err = np.abs(lower_bounds - cycle_life)
                upper_err = np.abs(upper_bounds - cycle_life)
                pred_dict[model][model+' Predicted cycle life'] = cycle_life
                pred_dict[model][model+' Predicted CL error'] = [[lower_err[i], upper_err[i]] for i in range(len(lower_err))]
        pred_df = reduce(lambda x, y: pd.merge(x, y, on = 'Name'),[pred_dict[model] for model in self.ml_model])
        if train_test == 'train':
            pred_df['Actual cycle life'] = np.power(10, self.y_train)
        elif train_test == 'test':
            pred_df['Actual cycle life'] = np.power(10, self.y_test)
        return pred_df