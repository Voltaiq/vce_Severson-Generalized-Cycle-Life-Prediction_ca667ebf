import ipywidgets as widgets
from ipywidgets import interactive, fixed
import severson_featurization
import importlib
importlib.reload(severson_featurization)
# from severson_featurization import calc_X_and_y, drop_unfinished_tests

std_train_datasets = ['Severson2019 - All (LFP)','Severson2019 - Train (LFP)',
                      'Severson2019 - Test (LFP)','Severson2019 - Test2 (LFP)','Attia2020 (LFP)',
                      'Devie2018 (NMC/LCO)','Wei2011 (LCO)','Juarez-Robles2020 (NCA)',
                      'Weng2021 (NMC)','Custom']


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
    dataset_lookup['Wei2011 (LCO)'] = severson_featurization.filter_test_record_by_one_name("CALCE", trs)
    dataset_lookup['Juarez-Robles2020 (NCA)'] = severson_featurization.filter_test_record_by_one_name(
        "UL-PUR", trs)
    dataset_lookup['Weng2021 (NMC)'] = severson_featurization.filter_test_record_by_name(
        ["UM","Cycling"], trs)

    return dataset_lookup

def populate_test_train_data(prediction_object, trs, predict = False):
    ''' function to populate all training and testing data for the given prediction_object based on the train and test lists of that object'''
    dataset_lookup = get_standard_datasets(trs)
    
    if not predict:
        train_list_datasets = prediction_object.get_train_test_list('train')
        train_list_custom = [severson_featurization.filter_test_record_by_one_name(name, trs) for name in prediction_object.get_custom_train_test_list('train')]
        train_list = [item for sublist in train_list_custom for item in sublist]

        for dataset in train_list_datasets:
            train_list.extend(dataset_lookup[dataset])

        prediction_object.populate_train_test_records(train_list)

        if not prediction_object.get_train_test_split_option():
            test_list_datasets = prediction_object.get_train_test_list('test')
            test_list_custom = [severson_featurization.filter_test_record_by_one_name(name, trs) for name in prediction_object.get_custom_train_test_list('test')]
            test_list = [item for sublist in test_list_custom for item in sublist]

            for dataset in test_list_datasets:
                test_list.extend(dataset_lookup[dataset])

            prediction_object.populate_test_test_records(test_list)
            
    else:
        predict_list_datasets = prediction_object.get_train_test_list('predict')
        
        pred_list_custom = [severson_featurization.filter_test_record_by_one_name(name, trs) for name in prediction_object.get_custom_train_test_list('predict')]
        pred_list = [item for sublist in pred_list_custom for item in sublist]

        for dataset in predict_list_datasets:
            pred_list.extend(dataset_lookup[dataset])

        prediction_object.populate_predict_test_records(pred_list)


def select_custom_tests(options,train_or_test, prediction_obj):
    ''' function to select custom tests from a list of options, setting the list to the prediction object passed in'''
    if len(options) > 0:
        if 'All' in options:
            prediction_obj.replace_train_test_list_custom(prediction_obj.get_custom_search_results(),
                                                          train_or_test)
        else:
            prediction_obj.replace_train_test_list_custom(options,train_or_test)
        print("custom: ", prediction_obj.get_custom_train_test_list(train_or_test))
        
def custom_select(other_search_text, train_or_test, prediction1, trs):
    # search for tests containing other_search_text
    test_list = []
    if other_search_text != '':
        if other_search_text != prediction1.get_last_custom_search():
            search_changed = True
        else:
            search_changed = False
        prediction1.set_last_custom_search(other_search_text)
        stitch = severson_featurization.filter_test_record_by_one_name(other_search_text,trs)
        
        for test in stitch:
            test_list.append(test.name)
        print(str(len(test_list)-1) + " tests found")
        
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
        selected_tests = interactive(select_custom_tests, options = widgets.SelectMultiple(
            value = selection_values, options=test_list, description=f'Select tests:',
            style={'description_width': 'initial'}, ensure_option=True),
                                     train_or_test=fixed(train_or_test),prediction_obj = fixed(prediction1))
        display(selected_tests)
        
def select_widget(train_sets, train_or_test,pred_obj, trs, predict_button = None):
    train_list = list(train_sets)
    if len(train_list ) > 0:
        if train_or_test == 'predict':
            predict_button.disabled = False
        if 'Custom' in train_list:
            print("Custom test search:")
            other_train = interactive(custom_select, other_search_text = widgets.Text(
                value = pred_obj.get_last_custom_search(),description='Test name search:', 
                style={'description_width': 'initial'},continuous_update=False),
                                     train_or_test=fixed(train_or_test), prediction1 = fixed(pred_obj),
                                     trs = fixed(trs))
            display(other_train)
            train_list.remove('Custom')
            
        if 'Custom' not in train_sets:
            # need to remove fields from the custom list
            pred_obj.replace_train_test_list_custom((),train_or_test)
        pred_obj.set_train_test_list(train_list,train_or_test)
        print(train_or_test, " datasets selected: ", pred_obj.get_train_test_list(train_or_test))
        
def train_test_ratio(ratio, pred_obj):
    ''' function to set the train/test ratio of sklearn's train-test-split within the prediction class'''
    pred_obj.set_train_test_split_ratio(ratio)
    
def test_select_method(method, prediction1, trs):
    ''' function for initiating the test dataset selection method'''
    if method == 'Use train_test_split on training dataset':
        prediction1.set_train_test_split_option(True)
        train_test_split_ratio = interactive(train_test_ratio, 
                                             ratio = widgets.BoundedFloatText(value=0.6,min=0,max=1.0,step=0.05,description='Train-test split ratio:',style={'description_width': 'initial'},disabled=False),
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
        
        
def featurize_inputs_widget(start_cycle, end_cycle, per_cap_ret, prediction1):
    ''' widget function to pass values to the prediction object'''
    prediction1.set_start_cycle(start_cycle)
    prediction1.set_end_cycle(end_cycle)
    prediction1.set_cap_retention_threshold(per_cap_ret)