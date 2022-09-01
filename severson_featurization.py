# imports
import voltaiq_studio as vs
from voltaiq_studio import TraceFilterOperation
import numpy as np
import pandas as pd
from functools import reduce
import scipy
from scipy.interpolate import interp1d


# filter test record functions
def filter_test_record_by_name(names, trs):
    ''' 
    Filter test records by a list of names.
    Return list of test record objects
    '''
    return [t for t in trs if all(x.lower() in t.name.lower() for x in names)] 

def filter_test_record_by_one_name(name, trs):
    ''' 
    Filter test records by a single keyword/name.
    Return list of test record objects
    '''
    return [t for t in trs if name.lower() in t.name.lower()] 

def filter_tr_by_name_exclusion(name,exclude, trs):
    ''' 
    Filter test records by a single keyword/name. 
    Exclude any test records from the list which are in the exclude list.
    Return list of test record objects
    '''
    return [t for t in trs if (name.lower() in t.name.lower() and all(x.lower() not in t.name.lower() for x in exclude))] 


# load data from each test record for the given cycles:

def load_data(test,cyc_1, cyc_2):
    ''' 
    Load time-series data for a test record, specifically the voltage, discharge capacity, test time, and cycle number traces.
    Only load data for cycles cyc_1 and cyc_2.
    Return a pandas dataframe
    '''
#     test.clear_cache()
    reader = test.make_time_series_reader()
    reader.add_trace_keys('h_potential','h_discharge_capacity','h_test_time', 'h_current')
    reader.filter_trace('h_current', TraceFilterOperation.LESS_THAN, 0)
    reader.add_info_keys('i_cycle_num')
    reader.filter_cycle_list([cyc_1, cyc_2]) 

    return reader.read_pandas()

def interpolate_data(df, name):
    ''' 
    Interpolates discharge capacity data between VDlin values (currently set to 3.6V - 2V).
    
    Returns: dataframe with interpolated column 'Interp Dis Q' and cycle number 'Cyc Num'
    '''
    max_voltage = np.nanmax(df['h_potential'])
    min_voltage = np.nanmin(df['h_potential'])
    voltage_linspace = np.linspace(max_voltage-max_voltage*.001, min_voltage+min_voltage*.01, 1000)
    temp = df.drop(columns=['h_current','h_test_time'])
    QDint = []
    cyc_int = []
    for cyc in pd.unique(temp['i_cycle_num']):
        x = temp[temp['i_cycle_num']==cyc]['h_potential']
        y = temp[temp['i_cycle_num']==cyc]['h_discharge_capacity']
        if name == '2017-06-30_5_2C-71per_3C_CH36_VDF' and cyc == 99: # this test has an erroneous point at the start, coming from the charge region
            x.reset_index(drop=True)[1:]
            y.reset_index(drop=True)[1:]
        f = interp1d(x,y, fill_value="extrapolate")
        ynew = f(voltage_linspace)
        QDint.extend(ynew)
        cyc_int.extend([cyc]*len(ynew))
    df_int = pd.DataFrame()
    df_int['Interp Dis Q'] = QDint
    df_int['Cyc Num'] = cyc_int
    return df_int

def delta_QV(df,start, end):
    '''
    Calculates the change in the capacity (V) between the start and end cycle
    Parameters: start = start cycle, end = end cycle
    
    returns: dataframe of delta capacity values
    '''
    
    return df[df['Cyc Num']==end].reset_index()['Interp Dis Q'] - df[df['Cyc Num']==start].reset_index()['Interp Dis Q']

def delta_Q_lowV(df,start, end):
    ''' 
    Calculates the difference in the capacity at the lowest voltage point between the start cycle and end cycle
    Returns a number
    '''
    # the end voltage point is always the last one, defined by Vdlin
    q_start = df[df['Cyc Num']==start]['Interp Dis Q'][999]
    q_end = df[df['Cyc Num']==end]['Interp Dis Q'][1999]
    return q_end - q_start

def slope_intercept(df,start,end):
    ''' 
    Takes a dataframe containing cycle number and discharge capacity columns and performs
    linear regression fit from cycle 'start' to cycle 'end' 
    
    returns: z = [slope, intercept] of linear regression
    '''
    x = np.array(df[df['cycle_number'].isin(np.arange(start,end,1))]['cycle_number'])
    y = np.array(df[df['cycle_number'].isin(np.arange(start,end,1))]['cyc_discharge_capacity'])
    z = np.polyfit(x,y,1)
    return z

def cyc_end_of_life(df,name,cap_percent, predict):
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
    
    
    df_eol = df.copy()
    # ref_cycle is the reference cycle (index) that should be chosen for normalization. Default is to use the second logged cycle
    ref_cycle = df_eol.cycle_number.iloc[1]
    cap_initial = df_eol['cyc_discharge_capacity'].iloc[1]
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
            end_cyc = indices_below_cap_percent[consecutive_cycles[0]] -3
            if end_cyc <0:
                end_cyc = indices_below_cap_percent[consecutive_cycles[0]]
        else:       
            print(f"Test {name} does not reach {cap_percent*100}%. Second-to-last cycle capacity retention: {df_eol['Capacity retention'].iloc[-2]*100}%")
    else:
        if not predict:
            print(f"Test {name} does not reach {cap_percent*100}%. Second-to-last cycle capacity retention: {df_eol['Capacity retention'].iloc[-2]*100}%")
    return end_cyc   


def calc_X_and_y(tr_list,cycle_start, cycle_end, cap_percent, predict = False,trs = None):
    ''' 
    From a list of test records, return a dataframe X of all calculated features 
    and a dataframe y containing the log cycle life values
    
    cycle life defined by cap_percent which is a percent capacity rentention
    predict: if True, will not calculate a y dataframe
    
    returns: dataframes X, y
    '''
    # initialize lists for each feature calculation
    device_name = []
    
    #discharge cap difference related
    min_deltaQ = []
    mean_deltaQ = []
    var_deltaQ = []
    skew_deltaQ = []
    kur_deltaQ = []
    deltaQ_lowV = []

    # other discharge
    slope_2_100 = []
    intcpt_2_100 = []
    slope_91_100 = []
    intcpt_91_100 = []
    q_2 = []
    maxQ_q2 = []
    q_100 = []
    total_cycles = []
    avg_cyc_time = []

    y_cyc_life = []
    
    cyc_dict = {'2017-05-12_3_6C-80per_3_6C_CH2_VDF': '2017-06-30_3_6C-80per_3_6C_CH2_VDF', '2017-05-12_4C-80per_4C_CH5_VDF': '2017-06-30_4C-80per_4C_CH5_VDF',
                  '2017-05-12_3_6C-80per_3_6C_CH1_VDF': '2017-06-30_3_6C-80per_3_6C_CH1_VDF', '2017-05-12_3_6C-80per_3_6C_CH3_VDF': '2017-06-30_3_6C-80per_3_6C_CH3_VDF',
                  '2017-05-12_4C-80per_4C_CH6_VDF': '2017-06-30_4C-80per_4C_CH6_VDF'}

    for t in tr_list:
        # calculate time-series dependend statistics

        data = load_data(t,cycle_start, cycle_end)

        interp_data = interpolate_data(data, t.name)

        deltaQ = delta_QV(interp_data, cycle_start, cycle_end)

        device_name.append(t.name)
        min_deltaQ.append(np.log10(abs(np.nanmin(deltaQ))))
        mean_deltaQ.append(np.log10(abs(np.mean(deltaQ))))
        var_deltaQ.append(np.log10(abs(np.var(deltaQ))))
        skew_deltaQ.append(np.log10(abs(scipy.stats.skew(deltaQ,nan_policy='omit'))))
        kur_deltaQ.append(np.log10(abs(scipy.stats.kurtosis(deltaQ,nan_policy='omit'))))
        deltaQ_lowV.append(delta_Q_lowV(interp_data, cycle_start, cycle_end))

        # if one of 5 tests that were not completed in batch1, then need to stitch the cycle stats together.
        if t.name in cyc_dict.keys():
            cycle1 = t.get_cycle_stats()[['cyc_discharge_capacity','cycle_number','cyc_total_cycle_time']]
            second_test = cyc_dict[t.name]
            # get test record object for second_test
            cycle2 = filter_test_record_by_one_name(second_test,trs)[0].get_cycle_stats()[['cyc_discharge_capacity','cycle_number','cyc_total_cycle_time']]
            # stitch the relevant data together - cycle needs to increment
            
            segment_start_cycle = cycle1.cycle_number.iloc[-1] + 1
            adjust_cyc_num = segment_start_cycle - cycle2.cycle_number.iloc[0]
            cycle2.cycle_number = cycle2.cycle_number + adjust_cyc_num
            cycle = pd.concat([cycle1,cycle2], ignore_index = True)
            
        else:
            cycle = t.get_cycle_stats()

        if predict:
            # average cycle time in hours
            avg_cyc_time.append(np.nanmean(cycle.cyc_total_cycle_time)/3600) # might want to adjust this to ignore 0's and -#s

        cyc2 = slope_intercept(cycle,1,cycle_end) # we are zero-indexed instead of 1-indexed
        cyc91 = slope_intercept(cycle,90,cycle_end) # we are zero-indexed instead of 1-indexed

        slope_2_100.append(cyc2[0])
        intcpt_2_100.append(cyc2[1])
        slope_91_100.append(cyc91[0])
        intcpt_91_100.append(cyc91[1])
        if 'batch9' in t.name:
            cyc2_num = 3
        else:
            cyc2_num = 1
        q_2.append(float(cycle[cycle['cycle_number']==cyc2_num]['cyc_discharge_capacity'])) # we are zero-indexed instead of 1-indexed
        maxQ_q2.append(max(cycle['cyc_discharge_capacity']) - float(cycle[cycle['cycle_number']==cyc2_num]['cyc_discharge_capacity'])) # we are zero-indexed instead of 1-indexed
        q_100.append(float(cycle[cycle['cycle_number']==cycle_end]['cyc_discharge_capacity']))

        cyc_life = cyc_end_of_life(cycle,t.name,cap_percent, predict)

        if cyc_life == 0:
            y_cyc_life.append("unfinished")
        else:
            if not predict:
                y_cyc_life.append(np.log10(cyc_life))
            else:
                y_cyc_life.append(cyc_life)
        
        total_cycles.append(t.total_cycles)

    
    X = pd.DataFrame()
    
    X['min_deltaQ'] = min_deltaQ
    X['mean_deltaQ'] = mean_deltaQ
    X['var_deltaQ'] = var_deltaQ
    X['skew_deltaQ'] = skew_deltaQ
    X['kur_deltaQ'] = kur_deltaQ
    X['deltaQ_lowV'] = deltaQ_lowV

    # other discharge
    X['slope_2_100'] = slope_2_100
    X['intcpt_2_100'] = intcpt_2_100
    X['slope_91_100'] = slope_91_100
    X['intcpt_91_100'] = intcpt_91_100
    X['q_2'] = q_2
    X['maxQ_q2'] = maxQ_q2
    X['q_100'] = q_100

    X['Name'] = device_name

    if not predict:
        y = pd.DataFrame(y_cyc_life,columns = ['log_cyc_life'])
    else:
        y = pd.DataFrame(y_cyc_life,columns = ['cyc_life'])

    if not predict:
        return X, y
    else:
        return X, pd.DataFrame(total_cycles, columns = ['total_cycles']), y, avg_cyc_time

    
def drop_unfinished_tests(X, y):
    '''
    Takes an X feature dataframe and a y label dataframe and drops rows where the y-value is -inf (corresponds to cycle life of zero)
    
    Returns:
    X and y with only tests that reach the discharge capacity criteria
    Prints:
    Number of tests that were dropped
    '''
    drop_rows = list(y[y['log_cyc_life']=='unfinished'].index)
    print("number of tests dropped:",len(drop_rows))
    return X.drop(drop_rows).reset_index(drop=True), y.drop(drop_rows).reset_index(drop=True)
