import ipywidgets as widgets
from ipywidgets import interactive, fixed
import severson_featurization
import importlib
importlib.reload(severson_featurization)
import pandas as pd
from IPython.display import display, Markdown
# from severson_featurization import calc_X_and_y, drop_unfinished_tests

std_train_datasets = ['Severson2019 - All (LFP)','Severson2019 - Train (LFP)',
                      'Severson2019 - Test (LFP)','Severson2019 - Test2 (LFP)','Attia2020 (LFP)',
                      'Devie2018 (NMC/LCO)','He2011 (LCO)','Juarez-Robles2020 (NCA)',
                      'Weng2021 (NMC)','Mohtat2021 (NMC)','Custom']


def get_standard_datasets(trs):
    # Severson train_test_test2 split
    # list of names to filter
    dates = ['2017-05-12','2017-06-30']
    train_channels_512 = ['CH2_','CH5_','CH7_','CH9_','CH14_','CH16_',
                          'CH24_','CH18_','CH12_','CH25_','CH33_','CH27_','CH35','CH29','CH39',
                          'CH37','CH41','CH43','CH45','CH47']
    train_channels_630 = ['CH9_','CH11_','CH13_','CH15_','CH27_','CH29_',
                          'CH17_','CH19_','CH21','CH23','CH7_','CH24','CH26','CH32','CH34',
                          'CH36','CH38','CH40','CH42','CH44','CH46']
    train = []
    for ch in train_channels_512:
        train.append([dates[0],ch])
    for ch in train_channels_630:
        train.append([dates[1],ch])

    test_channels_512 = ['CH1_','CH3_','CH6_','CH10_','CH20','CH15_','CH23_',
                         'CH17_','CH11_','CH32_','CH26_','CH34_','CH28_','CH36_','CH30_','CH40_',
                         'CH38_','CH42_','CH44_','CH46_','CH48_']
    test_channels_630 = ['CH10_','CH12_','CH14_','CH16_','CH28_','CH30','CH18_',
                         'CH20_','CH22_','CH4_','CH8_','CH25_',
                         'CH31_','CH33_','CH35_','CH37_','CH39_','CH41_','CH43_','CH45_','CH47_','CH48_']
    test = []
    for ch in test_channels_512:
        test.append([dates[0],ch])
    for ch in test_channels_630:
        test.append([dates[1],ch])

    test2_names = 'batch8'
    remove_test2 = ['2018-04-12_batch8_CH33','2018-04-12_batch8_CH46','2018-04-12_batch8_CH7',
                    '2018-04-12_batch8_CH41','2018-04-12_batch8_CH6','2018-04-12_batch8_CH12']
    
    tr_list_train = [severson_featurization.filter_test_record_by_name(name, trs) for name in train]
    tr_list_train_flat = [item for sublist in tr_list_train for item in sublist]
    tr_list_test = [severson_featurization.filter_test_record_by_name(name, trs) for name in test]
    tr_list_test_flat = [item for sublist in tr_list_test for item in sublist]
    tr_list_test2 = severson_featurization.filter_tr_by_name_exclusion(test2_names, remove_test2, trs)

    dataset_lookup = {}
    dataset_lookup['Severson2019 - All (LFP)'] = tr_list_train_flat + tr_list_test_flat + tr_list_test2
    dataset_lookup['Severson2019 - Train (LFP)'] = tr_list_train_flat
    dataset_lookup['Severson2019 - Test (LFP)'] = tr_list_test_flat
    dataset_lookup['Severson2019 - Test2 (LFP)'] = tr_list_test2
    dataset_lookup['Attia2020 (LFP)'] = severson_featurization.filter_test_record_by_one_name("batch9",
                                                                                              trs)
    dataset_lookup['Devie2018 (NMC/LCO)'] = severson_featurization.filter_test_record_by_name(
        ["HNEI","timeseries"], trs)
    dataset_lookup['He2011 (LCO)'] = severson_featurization.filter_test_record_by_one_name("CALCE", trs)
    dataset_lookup['Juarez-Robles2020 (NCA)'] = severson_featurization.filter_test_record_by_one_name(
        "UL-PUR", trs)
    dataset_lookup['Weng2021 (NMC)'] = severson_featurization.filter_test_record_by_name(
        ["UM","Cycling"], trs)
    dataset_lookup['Mohtat2021 (NMC)'] = severson_featurization.filter_test_record_by_name(['Cell_Expansion','cycling'],trs)

    return dataset_lookup

def populate_test_train_data(prediction_object, trs, predict = False):
    ''' function to populate all training and testing data for the given prediction_object based on the train and test lists of that object'''
    dataset_lookup = get_standard_datasets(trs)
    
    if not predict:
        train_list_datasets = prediction_object.get_train_test_list('train')
        train_list_custom = [severson_featurization.filter_test_record_by_one_name(name, trs) for name in prediction_object.get_custom_train_test_list('train')]
        train_list = [item for sublist in train_list_custom for item in sublist]
        train_dataset_group_list = ['custom']*len(train_list)

        for dataset in train_list_datasets:
            train_list.extend(dataset_lookup[dataset])
            train_dataset_group_list.extend([dataset]*len(dataset_lookup[dataset]))

        prediction_object.populate_train_test_records(train_list,train_dataset_group_list)

        if not prediction_object.get_train_test_split_option():
            test_list_datasets = prediction_object.get_train_test_list('test')
            test_list_custom = [severson_featurization.filter_test_record_by_one_name(name, trs) for name in prediction_object.get_custom_train_test_list('test')]
            test_list = [item for sublist in test_list_custom for item in sublist]
            test_dataset_group_list = ['custom']*len(test_list)

            for dataset in test_list_datasets:
                test_list.extend(dataset_lookup[dataset])
                test_dataset_group_list.extend([dataset]*len(dataset_lookup[dataset]))

            prediction_object.populate_test_test_records(test_list,test_dataset_group_list)
            
    else:
        predict_list_datasets = prediction_object.get_train_test_list('predict')
        
        pred_list_custom = [severson_featurization.filter_test_record_by_one_name(name, trs) for name in prediction_object.get_custom_train_test_list('predict')]
        pred_list = [item for sublist in pred_list_custom for item in sublist]
        pred_dataset_group_list = ['custom']*len(pred_list)

        for dataset in predict_list_datasets:
            pred_list.extend(dataset_lookup[dataset])
            pred_dataset_group_list.extend([dataset]*len(dataset_lookup[dataset]))

        prediction_object.populate_predict_test_records(pred_list,pred_dataset_group_list)


# def select_custom_tests(options,train_or_test, prediction_obj):
#     ''' function to select custom tests from a list of options, setting the list to the prediction object passed in'''
#     if len(options) > 0:
#         if 'All' in options:
#             prediction_obj.replace_train_test_list_custom(prediction_obj.get_custom_search_results(),
#                                                           train_or_test)
#         else:
#             prediction_obj.replace_train_test_list_custom(options,train_or_test)
#         print("custom: ", prediction_obj.get_custom_train_test_list(train_or_test))
        
def select_custom_tests(options,train_or_test, prediction_obj,remove):
    ''' function to select custom tests from a list of options, setting the list to the prediction object passed in'''
    if len(options) > 0:
        if 'All' in options:
            if remove:
                prediction_obj.remove_train_test_list_custom(prediction_obj.get_custom_search_results(),train_or_test)
            else:
                prediction_obj.replace_train_test_list_custom(prediction_obj.get_custom_search_results(),
                                                          train_or_test)
        else:
            if remove:
                prediction_obj.remove_train_test_list_custom(options,train_or_test)
            else:
                prediction_obj.replace_train_test_list_custom(options,train_or_test)
        print(str(len(prediction_obj.get_custom_train_test_list(train_or_test))) + " tests selected")
        
def clear_all_tests(train_or_test, pred_obj):
    pred_obj.clear_train_test_list_custom(train_or_test)
    print("Removed all custom data from the " + train_or_test + " dataset")

def filter_by_cycle_num(cyc_num,trs_list):
    return [t for t in trs_list if t.total_cycles >= cyc_num]
 
    
def cap_ret_reached(t,cap_percent,pred_obj,ref_cyc):
    ''' 
    Calculates the cycle at which the test drops below cap_percent % of initial capacity
    for the cycle-stats dataframe passsed to the function. Bases initial capacity on the second cycle of a test. 
    
    df - cycle-stats dataframe
    name - name of the test record
    cap_percent - percentage value which defines end of life.
    predict - Boolean stating whether this calculation is being performed on prediction data, where the expectation is that many tests might not have reached the cap_percent yet
    
    returns: cycle number at which cap_percent % of initial cycle capacity is reached    
    '''
    end_cyc = 0
    
#     print(t.name)
    
    
    df_eol = t.get_cycle_stats()[['cyc_discharge_capacity','cycle_number','cyc_total_cycle_time']]
    # ref_cycle is the reference cycle (index) that should be chosen for normalization
    try:
        cap_initial = float(df_eol[df_eol['cycle_number']==ref_cyc]['cyc_discharge_capacity'])
    except:
        print(f"Cycle {ref_cyc} does not exist for test {name}. Attempting to use cycle ordinal instead.")
        try:
            ref_cycle = df_eol.cycle_number.iloc[ref_cyc]
            cap_initial = df_eol['cyc_discharge_capacity'].iloc[ref_cyc]
            print(f"Using reference cycle {ref_cycle} for test {name}")
        except:
            print(f"No appropriate cycle could be chosen for test {name}. Aborting featurization.") 
            #could improve by just dropping this test...
            return False
#     try:
#         cap_initial = float(df_eol[df_eol['cycle_number']==ref_cyc]['cyc_discharge_capacity'])
#     except:
#         return False
    df_eol['Capacity retention'] = df_eol['cyc_discharge_capacity']/cap_initial
    
    # logic to calculate the cycle to cap_percent percent capacity. Can't just take the first value that matches, because could have a dip for other reasons.
    # look for 5 consecutive cycles that are below cap_percent %, then choose the first of those

    # series showing all indices where the capacity retention is less than cap_percent    
    indices_below_cap_percent = pd.Series(df_eol[df_eol['Capacity retention']<=cap_percent]['cycle_number'].index)
    # We want at least 5 consecutive data points below this capacity; this can only happen if there are at least 5 indices present
    if len(indices_below_cap_percent) > 5:
        # calculate the difference in indices - we want this difference to be 1 for 5 consecutive cycles
        indices_difference = indices_below_cap_percent.diff()
        # rolling sum provides information on the 5 consecutive rows of indices, placing the sum at the center index. 
        #We will want to subtract 3 from the final value to adjust for this 'center' calculation
        indices_difference_five = indices_difference.rolling(5, min_periods=1, center = True).sum()
        consecutive_cycles = indices_difference_five[indices_difference_five <= 5].index.drop(0, errors = 'ignore')
        if len(consecutive_cycles) > 0:
            return True
        else:       
            return False
    else:
        return False  


def filter_min_cap_retention(cap_retention,trs_list,pred_obj):
    return [t for t in trs_list if cap_ret_reached(t,cap_retention,pred_obj, pred_obj.reference_cycle)]
    
def custom_select(filter_by_cap_retention,min_cyc_num,other_search_text, train_or_test, prediction1, trs):
    # search for tests containing other_search_text
    test_list = []
    if other_search_text != '':
        if other_search_text != prediction1.get_last_custom_search():
            search_changed = True
        else:
            search_changed = False
        prediction1.set_last_custom_search(other_search_text)
        
        initial_test_list = severson_featurization.filter_test_record_by_one_name(other_search_text,trs)
        initial_test_list = filter_by_cycle_num(min_cyc_num,initial_test_list)
        if filter_by_cap_retention:
            initial_test_list = filter_min_cap_retention(prediction1.capacity_retention_threshold,
                                                         initial_test_list,pred_obj=prediction1)
        
        for test in initial_test_list:
            test_list.append(test.name)
        print(str(len(test_list)) + " tests found")
        
        if train_or_test == 'test':
            unique_test_list = [test for test in test_list if test not in 
                                prediction1.get_custom_train_test_list('train')]
            if len(unique_test_list) < len(test_list):
                print("Some test records already included in the training dataset. Removed " + 
                      str(len(test_list) - len(unique_test_list)) + " tests from search result.")
            test_list = unique_test_list
        
        prediction1.save_custom_search_results(test_list)
#     return test_list
        test_list = ['All'] + test_list
    # allow users to select the tests they'd like to include in the analysis
        if search_changed:
            selection_values = []
        else:
            selection_values = prediction1.get_custom_train_test_list(train_or_test)
        selected_tests = widgets.SelectMultiple(value = selection_values, options=test_list, description=f'Select tests:',
                                                style={'description_width': 'initial'}, ensure_option=True)
        
        add_tests_button = widgets.Button(description = 'Add tests', button_style = 'danger', style={"button_color": "#38adad"})
        remove_tests_button = widgets.Button(description = 'Remove tests', button_style = 'danger', style={"button_color": "#38adad"})
        clear_all_tests_button = widgets.Button(description = "Clear custom tests", button_style = 'danger', style = {"button_color": "#38adad"})
        
        output = widgets.Output()

        display(selected_tests, add_tests_button, remove_tests_button, clear_all_tests_button, output)

        def add_tst_button(b):
            with output:
                select_custom_tests(selected_tests.value, train_or_test=train_or_test,prediction_obj = prediction1,remove=False)



        def remove_tst_button(b):
            with output:
                select_custom_tests(selected_tests.value, train_or_test=train_or_test,prediction_obj = prediction1,remove=True)

        def clear_tst_button(b):
            with output:
                clear_all_tests(train_or_test, prediction1)
                
        add_tests_button.on_click(add_tst_button)
        remove_tests_button.on_click(remove_tst_button)
        clear_all_tests_button.on_click(clear_tst_button)
        
def select_widget(train_sets, train_or_test,pred_obj, trs, predict_button = None):
    train_list = list(train_sets)
    if len(train_list ) > 0:
        if train_or_test == 'predict':
            predict_button.disabled = False
        if 'Custom' in train_list:
            print("Custom test search:")
            other_train = interactive(custom_select, 
                                      filter_by_cap_retention = widgets.Checkbox(
                                          value=False,description='Filter tests by capacity retention threshold',
                                          style={'description_width': 'initial'}),
                                      min_cyc_num = widgets.IntText(description = 'Minimum # of cycles:',
                                                                    style={'description_width': 'initial'}, 
                                                                    value = pred_obj.get_end_cycle()+1),
                                      other_search_text = widgets.Text(value = pred_obj.get_last_custom_search(),
                                                                       description='Test name search:',
                                                                       style={'description_width': 'initial'},
                                                                       continuous_update=False),train_or_test=fixed(train_or_test),
                                      prediction1 = fixed(pred_obj),trs = fixed(trs))  
#             custom_select, other_search_text = widgets.Text(
#                 value = pred_obj.get_last_custom_search(),description='Test name search:', 
#                 style={'description_width': 'initial'},continuous_update=False),
#                                      train_or_test=fixed(train_or_test), prediction1 = fixed(pred_obj),
#                                      trs = fixed(trs))
        
            display(other_train)
            train_list.remove('Custom')
            
        if 'Custom' not in train_sets:
            # need to remove fields from the custom list
            pred_obj.replace_train_test_list_custom((),train_or_test)
        pred_obj.set_train_test_list(train_list,train_or_test)
        print(train_or_test, " datasets selected: ", pred_obj.get_train_test_list(train_or_test))
        
def train_test_ratio(ratio,pred_obj):
    ''' function to set the train/test ratio of sklearn's train-test-split within the prediction class'''
    pred_obj.set_train_test_split_ratio(ratio)
    
def test_select_method(method, prediction1, trs):
    ''' function for initiating the test dataset selection method'''
    if method == 'Use train_test_split on training dataset':
        prediction1.set_train_test_split_option(True)
        train_test_split_ratio = interactive(train_test_ratio, 
                                             ratio = widgets.BoundedFloatText(value=0.4,min=0,max=1.0,step=0.05,description='Train-test split ratio (test size):',style={'description_width': 'initial'},disabled=False),
                                            pred_obj = fixed(prediction1))
        display(train_test_split_ratio)
    elif method == 'Select test dataset manually':
        prediction1.set_train_test_split_option(False)
        std_test_datasets = [dataset for dataset in std_train_datasets if dataset not in prediction1.get_train_test_list('train')]
#         if ('Severson2019 - All (LFP)' in prediction1.get_train_test_list('train')):
#             std_test_datasets.remove(
#         if 'Severson2019 - Train (LFP)' in prediction1.get_train_test_list('train') or
#             'Severson2019 - Test (LFP)' in prediction1.get_train_test_list('train') or 'Severson2019 - Test2 (LFP)' in prediction1.get_train_test_list('train')):
            
        select_test = interactive(select_widget, 
                           train_sets = widgets.SelectMultiple(value=[], options=std_test_datasets, description=f'Test Datasets:',style={'description_width': 'initial'}, ensure_option=True),
                          train_or_test=fixed('test'), pred_obj = fixed(prediction1), trs = fixed(trs),
                                 predict_button = fixed(None))
        display(select_test)
        # likely want to have the same workflow as setting the training dataset, except that anything that was included in the training dataset should not be included in the testing dataset
        # possibly do a validation on that somehow... will be hardest for the custom option
        
        
def featurize_inputs_widget(start_cycle, end_cycle, per_cap_ret, prediction1,ref_cyc):
    ''' widget function to pass values to the prediction object'''
    prediction1.set_start_cycle(start_cycle)
    prediction1.set_end_cycle(end_cycle)
    prediction1.set_cap_retention_threshold(per_cap_ret)
    prediction1.set_reference_cycle(ref_cyc)