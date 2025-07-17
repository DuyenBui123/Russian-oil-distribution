# %% Load in library
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
from Code import data_preprocessing as pp

import datetime
import sys
import statistics as stats

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# %% 

os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
# import excel files with multiple sheets
shipfiles = pd.read_excel("./Data/Port Calls - 2024 - Tankers - RU & NL.xlsx", sheet_name = None)
print(type(shipfiles))
for sheet, df in shipfiles.items():
    print(sheet, type(df))
# creat empty dataframe to hold all loaded sheets
allshipfiles = pd.DataFrame()
# Iterate through daaframes in dictionary
for sheet_name, frame in shipfiles.items():
    # add column so we know which df data is from
    frame["PATH"] = sheet_name
    # add the dataframe to athe allshipfiles
    allshipfiles = pd.concat([allshipfiles, frame])
    
    
# Create directory to story data  
datapath_nested_directory = './preprocessing/pp_inter_input' 
try: 
    os.makedirs(datapath_nested_directory)
    print(f"Nested directory' '{datapath_nested_directory}' created successfully")
except FileExistsError:
    print(f"One or more direcotries in '{datapath_nested_directory}' aldready exist")
    
except PermissionError():
    print(f"Permission denied: Unable to create '{datapath_nested_directory}")
except Exception as e:
    print(f"An error occured: {e}")

    
allshipfiles.to_csv('./preprocessing/inter_input/All Port calls - NL & RU.csv', index = False)


  # %% extract df with only tanker went to NL  
tanker_went_to_NL = allshipfiles[allshipfiles.PATH == 'Tankers went to NL']
tanker_went_to_NL.dtypes
pd.set_option("display.max_columns", None)
print(tanker_went_to_NL.head())
print(tanker_went_to_NL['IMO'].count())
# count unique IMO numbers
len(set(tanker_went_to_NL['IMO']))
# extract unique and counts of each IMO
IMO, counts = np.unique(tanker_went_to_NL['IMO'], return_counts = True)
D2_IMO_np = np.vstack((IMO, counts)).T
# convert all unique and counts array into dataframe
IMO_df = pd.DataFrame(D2_IMO_np, columns=['IMO', 'counts'])
# extract unique IMO based on a certain condition
IMO_max_pathes = IMO_df[IMO_df.counts == IMO_df.counts.max()]
IMO_path_larger_10 = IMO_df[IMO_df.counts >10]
# extract a specific IMO detail data
tanker_went_to_NL_maxpath = tanker_went_to_NL[tanker_went_to_NL['IMO'] == IMO_max_pathes.loc[614,'IMO']]
tanker_went_to_NL_9273064 = tanker_went_to_NL[tanker_went_to_NL['IMO'] == 9273064]                       
tanker_went_to_NL_9350460 = tanker_went_to_NL[tanker_went_to_NL['IMO'] == 9350460]

# group by portname and lastportofcall
seconds_in_day = 24*60*60
tanker_went_to_NL_9350460['DIFFTIME'] = tanker_went_to_NL_9350460['ARRIVALDATE']-tanker_went_to_NL_9350460['LASTPORTOFCALLSAILDATE']
# calculate return date for two consecutive trips

tanker_went_to_NL_9350460['RETURNDATE'] = tanker_went_to_NL_9350460['ARRIVALDATE']-tanker_went_to_NL_9350460['SAILDATE'].shift(1)


tanker_went_to_NL_9350460['DIFFTIME_HR'] = round((tanker_went_to_NL_9350460['DIFFTIME'].dt.days * seconds_in_day + tanker_went_to_NL_9350460['DIFFTIME'].dt.seconds)/3600,2)
# create a df with counting the frequency of different groups
tanker_went_to_NL_9350460_count = tanker_went_to_NL_9350460.groupby(['PORTNAME','LASTPORTOFCALL', 'LASTPORTOFCALLCTRY']).size().reset_index(name='COUNT')
tanker_went_to_NL_9350460_count['NEW_INDEX'] = tanker_went_to_NL_9350460_count.index
tanker_went_to_NL_9350460_merged = pd.merge(tanker_went_to_NL_9350460, tanker_went_to_NL_9350460_count, on = ['PORTNAME','LASTPORTOFCALL', 'LASTPORTOFCALLCTRY'], how = 'inner')

# %% inspecct tankers were to Netherlands (WW)

tanker_were_to_NL_ww = allshipfiles[allshipfiles.PATH == 'Tankers were to NL (worldwide)']
tanker_were_to_RU_ww = allshipfiles[allshipfiles.PATH == 'Tankers were to RU (worldwide)']
tanker_went_to_NL = allshipfiles[allshipfiles.PATH == 'Tankers were to NL']
tanker_went_to_RU = allshipfiles[allshipfiles.PATH == 'Tankers were to RU']
# check dublicated IMO in RU and NL dataset
NL_domes_imo = tanker_went_to_NL['IMO'].unique()
RU_domes_imo = tanker_went_to_RU['IMO'].unique()
NL_imo = tanker_were_to_NL_ww['IMO'].unique()
RU_imo = tanker_were_to_RU_ww['IMO'].unique()
check_NLvsRUimo = np.isin(NL_imo, RU_imo)
check_NLdomesvsNL = np.isin(NL_imo,NL_domes_imo)
len(check_NLdomesvsNL[check_NLdomesvsNL == True])
len(check_NLvsRUimo[check_NLvsRUimo == True])
check_RUvsNLimo = np.isin(RU_imo, NL_imo)
len(check_RUvsNLimo[check_RUvsNLimo == True])
identical_imo = np.intersect1d(NL_imo, RU_imo)
check_NLvsRUimo = np.isin(NL_imo, RU_imo)
check_RUdomesvsRU = np.isin(RU_imo,RU_domes_imo)
len(check_NLdomesvsNL[check_NLdomesvsNL == True])
# Check whether the identical IMO's have the same POC
tanker_were_to_NL_ww_9960978 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9960978]
tanker_were_to_NL_ww_9960978 = tanker_were_to_NL_ww_9960978.reset_index(drop=True)
tanker_were_to_RU_ww_9960978 = tanker_were_to_RU_ww[tanker_were_to_RU_ww['IMO'] == 9960978]
not_iden = 0
for imo in identical_imo:
    NL_imo_poc = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == imo]
    NL_imo_poc = NL_imo_poc.drop(columns = ['PATH'])
    NL_imo_poc = NL_imo_poc.reset_index(drop=True)
    
    RU_imo_poc = tanker_were_to_RU_ww[tanker_were_to_RU_ww['IMO'] == imo]
    RU_imo_poc = RU_imo_poc.drop(columns = ['PATH'])
    RU_imo_poc = RU_imo_poc.reset_index(drop=True)
    
    if NL_imo_poc.equals(RU_imo_poc) == False:
        not_iden = not_iden + 1
        print('IMO:', imo, 'is not the same in two dataset')
        
        
        
pd.set_option("display.max_columns", None)


# extract unique and counts of each IMO
IMO, counts = np.unique(tanker_were_to_NL_ww['IMO'], return_counts = True)
D2_IMO_np_NL_ww = np.vstack((IMO, counts)).T
# convert all unique and counts array into dataframe
IMO_NL_ww_df = pd.DataFrame(D2_IMO_np_NL_ww, columns=['IMO', 'counts'])
# extract unique IMO based on a certain condition
IMO_max_pathes_NL_ww = IMO_NL_ww_df[IMO_NL_ww_df.counts == IMO_NL_ww_df.counts.max()]
IMO_path_larger_10_NL_ww = IMO_NL_ww_df[IMO_NL_ww_df.counts >10]
# extract a specific IMO detail data
tanker_were_to_NL_ww_maxpath = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == IMO_max_pathes_NL_ww.loc[614,'IMO']]
tanker_were_to_NL_ww_9273064 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9273064]                       
tanker_were_to_NL_ww_9350460 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9350460]
tanker_were_to_NL_ww_9524475 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9524475]
tanker_were_to_NL_ww_9833931 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9833931]
tanker_were_to_NL_ww_9424273 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9424273]
tanker_were_to_NL_ww_8353568 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 8353568]
tanker_were_to_NL_ww_5322099 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 5322099]
tanker_were_to_NL_ww_9391945 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9391945]
tanker_were_to_NL_ww_9956018 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9956018]
tanker_were_to_NL_ww_9973640 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9973640]
tanker_were_to_NL_ww_9666742 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9666742]

# %% explore data for tankers were to NL (ww)
# check for NaN
tanker_were_to_NL_ww = pp.check_nan_duplicate(tanker_were_to_NL_ww)
# check number of IMO containing both bunkering tanker (oil) and , (inland waterway)
tanker_were_to_NL_ww['SHIPTYPE'].unique()
tanker_were_to_NL_ww['IMO'][tanker_were_to_NL_ww['SHIPTYPE'] == 'Bunkering Tanker (Oil)']
sh_type = ['Bunkering Tanker (Oil), Inland Waterways', 'Bunkering Tanker (Oil)']
same_shiptype = tanker_were_to_NL_ww['SHIPTYPE'].isin(sh_type).reset_index()
tanker_were_to_NL_ww[tanker_were_to_NL_ww['SHIPTYPE'].isin(sh_type)]
len(same_shiptype[same_shiptype['SHIPTYPE'] == True])

# Check for the standard form of IMO
pp.check_IMO(tanker_were_to_NL_ww)
# validate the order consistancy of arrival date, sail date, last port call sail date
pp.validate_timestamp_sign(tanker_were_to_NL_ww)

# check identical arrival date, sail date, last POC sail date

arrvl_sail_dup, IMO_freq_arrvl_sail_dup, arrvl_sail_lastPOC_dup, IMO_freq_arrvl_sail_lastPOC_dup = pp.check_duplicated_timestamp(tanker_were_to_NL_ww)
tanker_were_to_NL_ww_9522128 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9522128]
tanker_were_to_NL_ww_9522128 = tanker_were_to_NL_ww_9522128.drop(['HOURSINPORT', 'PATH', 'DUPLICATE_2col', 'DUPLICATE_3col', 'HR_DIFF_Perc', 'STAYDURATION', 'STAYDURATION_HR', 'DURATIONINPORT_TIME'], axis = 'columns' )
tanker_were_to_NL_ww_9522128['TIME_GAP'] = tanker_were_to_NL_ww_9522128['ARRIVALDATE'] - tanker_were_to_NL_ww_9522128['SAILDATE'].shift(1)
tanker_were_to_NL_ww_9522128['TIME_TRAVEL_SEC'] = (tanker_were_to_NL_ww_9522128['TIME_GAP'].dt.days * seconds_in_day + tanker_were_to_NL_ww_9522128['TIME_GAP'].dt.seconds)
number_small_time_gap = tanker_were_to_NL_ww_9522128['TIME_TRAVEL_SEC'][tanker_were_to_NL_ww_9522128['TIME_TRAVEL_SEC'] < 851382]
len(number_small_time_gap) + len(number_small_time_gap)



tanker_were_to_NL_ww_9774185 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9774185]
tanker_were_to_NL_ww_9774185 = tanker_were_to_NL_ww_9774185.drop(['HOURSINPORT', 'PATH', 'DUPLICATE_2col', 'DUPLICATE_3col', 'HR_DIFF_Perc', 'STAYDURATION', 'STAYDURATION_HR', 'DURATIONINPORT_TIME'], axis = 'columns' )
tanker_were_to_NL_ww_9774185['TIME_GAP'] = tanker_were_to_NL_ww_9774185['ARRIVALDATE'] - tanker_were_to_NL_ww_9774185['SAILDATE'].shift(1)
tanker_were_to_NL_ww_8337992 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 8337992]
tanker_were_to_NL_ww_copy_8337992 = tanker_were_to_NL_ww_copy[tanker_were_to_NL_ww_copy['IMO'] == 8337992]
tanker_were_to_NL_ww_copy['SHIPTYPE'].unique()
tanker_were_to_NL_ww_9524475 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9524475]
tanker_were_to_NL_ww_9524475 = tanker_were_to_NL_ww_9524475.drop(['HOURSINPORT', 'PATH', 'DUPLICATE_2col', 'DUPLICATE_3col', 'HR_DIFF_Perc', 'STAYDURATION', 'STAYDURATION_HR', 'DURATIONINPORT_TIME'], axis = 'columns' )
tanker_were_to_NL_ww_9524475['TIME_GAP'] = tanker_were_to_NL_ww_9524475['ARRIVALDATE'] - tanker_were_to_NL_ww_9524475['SAILDATE'].shift(1)
tanker_were_to_NL_ww_9252278 = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == 9252278]
tanker_were_to_NL_ww_9252278 = tanker_were_to_NL_ww_9252278.drop(['HOURSINPORT', 'PATH', 'DUPLICATE_2col', 'DUPLICATE_3col', 'HR_DIFF_Perc', 'STAYDURATION', 'STAYDURATION_HR', 'DURATIONINPORT_TIME'], axis = 'columns' )
tanker_were_to_NL_ww_9252278['TIME_GAP'] = tanker_were_to_NL_ww_9252278['ARRIVALDATE'] - tanker_were_to_NL_ww_9252278['SAILDATE'].shift(1)
arrvl_sail_dup['IMO'].unique()
# Calculate time travel between two ports for each ship
# Calculate dublicated previous sail data and the next arrival date for each ship-or too small time gap for traveling
count_small_time_gap_0 = 0
count_small_time_gap_smaller1hr = 0
for IMO_nr in tanker_were_to_NL_ww['IMO'].unique():
    tanker_nr = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == IMO_nr]
    tanker_nr['TIME_TRAVEL'] = tanker_nr['ARRIVALDATE'] - tanker_nr['SAILDATE'].shift(1)
    tanker_nr['TIME_TRAVEL_SEC'] = (tanker_nr['TIME_TRAVEL'].dt.days * seconds_in_day + tanker_nr['TIME_TRAVEL'].dt.seconds)
    # number_small_time_gap = tanker_nr['TIME_TRAVEL_SEC'][tanker_nr['TIME_TRAVEL_SEC'] < 600]
    # count_small_time_gap = count_small_time_gap + len(number_small_time_gap)
    too_small_time_gap_index = tanker_nr.index[tanker_nr['TIME_TRAVEL_SEC'] == 0].tolist()
    count_small_time_gap_0 = count_small_time_gap_0 + len(too_small_time_gap_index)
    too_small_time_gap_index_smaller1hr = tanker_nr.index[tanker_nr['TIME_TRAVEL_SEC'] < 3600].tolist()
    count_small_time_gap_smaller1hr = count_small_time_gap_smaller1hr + len(too_small_time_gap_index_smaller1hr)
    print("IMO",IMO_nr,"has this list of IMO containing too small travel time gap (less than one hour):", too_small_time_gap_index_smaller1hr)
    
# calculate return date for two consecutive trips and plot the hours in port of a IMO

tanker_were_to_NL_ww_9485629['LEAVEDATE'] = tanker_were_to_NL_ww_9485629['SAILDATE']-tanker_were_to_NL_ww_9485629['LASTPORTOFCALLSAILDATE'].shift(-1)
tanker_were_to_NL_ww_9485629['DIFFTIME_HR'] = round((tanker_were_to_NL_ww_9485629['LEAVEDATE'].dt.days * seconds_in_day + tanker_were_to_NL_ww_9485629['LEAVEDATE'].dt.seconds)/3600,2)



tanker_were_to_NL_ww_9522128['ARRIVALDATE'].dtypes()
tanker_were_to_NL_ww_9522128.plot.line(x = 'ARRIVALDATE', y = 'DURATIONINPORT_HR', rot=90,x_compat=True)
tanker_were_to_NL_ww_9683673.plot.line(x = 'ARRIVALDATE', y = 'DURATIONINPORT_HR', rot=90,x_compat=True)

# standadize port name
tanker_were_to_NL_ww['PORTNAME'] = tanker_were_to_NL_ww['PORTNAME'].map(lambda x: pp.standardize_port_name(x))

# standadize ship name
tanker_were_to_NL_ww['SHIPTYPE'] = tanker_were_to_NL_ww['SHIPTYPE'].map(lambda x: pp.standadize_ship_type(x))


# Number of IMO Vessels by Frequency of Port Calls
# Frequency of port call per IMO
IMO_w_portcall_freq_uniq_df = pp.IMO_freq_port_call(tanker_were_to_NL_ww).drop_duplicates()
# Contribution of ship types to the total frequency of each bin
pp.plot_NO_ships_per_freq(IMO_w_portcall_freq_uniq_df)

# HOURS IN PORT
# validate hour in port
tanker_were_to_NL_ww,large_diff_HRinPrt_actHr_list = pp.validate_hrinport(tanker_were_to_NL_ww)
# plot Hours in Port Frequency
pp.plot_hourinport_freq_for_IMO(tanker_were_to_NL_ww, 'Hours in Port Frequency (tankers were to NL-ww)')
tanker_were_to_NL_ww['DURATIONINPORT_HR'].quantile(q = 0.75)
tanker_were_to_NL_ww['DURATIONINPORT_HR'].quantile(q = 0.90)
# plot the number of IMO in each ship type
stacked_bar_shiptype = IMO_w_portcall_freq_uniq_df.groupby('SHIPTYPE').size().reset_index()
stacked_bar_shiptype.rename(columns = {0 : 'Nr.IMO'}, inplace = True)
stacked_bar_shiptype.rename( columns={0 :'Nr.IMO'}, inplace=True )
ax = stacked_bar_shiptype.plot.bar('SHIPTYPE', 'Nr.IMO')
ax.set_xlabel('SHIP TYPE')
ax.set_ylabel('Nr. IMO')
plt.title("Total number of IMO in each ship type")
# determine the relationship between tanker types and hour in port
# determine the relationship between tanker types and the number of port frequency
pp.plot_boxplot(tanker_were_to_NL_ww.sort_values(by = 'SHIPTYPE'),IMO_w_portcall_freq_uniq_df.sort_values(by = 'SHIPTYPE'))
# determine outliners of hours in port using std
for ship_type in tanker_were_to_NL_ww['SHIPTYPE'].unique().tolist():
    #tanker_were_to_NL_ww
    shiptype = tanker_were_to_NL_ww[tanker_were_to_NL_ww['SHIPTYPE'] == ship_type]
    std_shiptype_hrinport = stats.stdev(shiptype['DURATIONINPORT_HR'])
    mean_shiptype_hrinport = shiptype['DURATIONINPORT_HR'].mean(axis = 0)
    lower_bound = mean_shiptype_hrinport - 2 * std_shiptype_hrinport
    upper_bound = mean_shiptype_hrinport + 2 * std_shiptype_hrinport
    plt.figure(figsize = (12,6))
    plt.scatter(x = shiptype['ARRIVALDATE'], y = shiptype['DURATIONINPORT_HR'], label = 'Hours in port', color = 'blue', alpha = 0.6)
    plt.axhline(lower_bound, color = 'red', linestyle = '--', label = '-2 STD')
    plt.axhline(upper_bound, color = 'red', linestyle = '--', label = '+2 STD')
    plt.axhline(mean_shiptype_hrinport, color =  'green', linestyle = '--', label = 'mean')
    plt.fill_between(shiptype['ARRIVALDATE'], lower_bound, upper_bound, color = 'gray', alpha = 0.2, label= '±2 STD Region')
    plt.title(f'Outliners of hours in port for {ship_type}', fontsize = 30)
    plt.gca().set_xlabel('Arrival time', fontsize=20)
    plt.gca().set_ylabel('Hours in port', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# determine outliners of hours in port usning quantile
for ship_type in IMO_w_portcall_freq_uniq_df['SHIPTYPE'].unique().tolist():
    #tanker_were_to_NL_ww
    shiptype = IMO_w_portcall_freq_uniq_df[IMO_w_portcall_freq_uniq_df['SHIPTYPE'] == ship_type]
    if len(shiptype) <2:
        print(shiptype['SHIPTYPE'], 'has number of IMO smaller than 2')
        continue
    std_shiptype_hrinport = stats.stdev(shiptype['Frequency'])
    mean_shiptype_hrinport = shiptype['Frequency'].mean(axis = 0)
    median_shiptype_hrinport = shiptype['Frequency'].median(axis = 0)
    lower_bound = shiptype['Frequency'].quantile(q=0.25) - 1.5*std_shiptype_hrinport

    plt.figure(figsize = (12,6))
    plt.scatter(x = shiptype['IMO'], y = shiptype['Frequency'], label = 'Total number of call', color = 'blue', alpha = 0.6)
    plt.axhline(lower_bound, color = 'red', linestyle = '--', label = '-1.5 STD')

    plt.axhline(mean_shiptype_hrinport, color =  'green', linestyle = '--', label = 'mean')
    plt.fill_between(shiptype['IMO'], lower_bound, mean_shiptype_hrinport, color = 'gray', alpha = 0.2, label= '-1.5 STD Region')
    plt.title(f'Outliners of frequency of port calls for {ship_type}', fontsize = 30)
    plt.gca().set_xlabel('IMO', fontsize=20)
    plt.gca().set_ylabel('Total number of call', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Amount of IMO full used, part time used


period_used_IMO = pd.DataFrame(tanker_were_to_NL_ww['IMO'].unique(), columns = ['IMO'])

period_used_IMO['period_used'] = 'Nan'
period_used_IMO['range'] = 'Nan'
# calculate number of months a IMO were on sea
for each_IMO in tanker_were_to_NL_ww['IMO'].unique():
    IMO_info = tanker_were_to_NL_ww[tanker_were_to_NL_ww['IMO'] == each_IMO]
    mont_used = round(abs((((IMO_info.iloc[0, IMO_info.columns.get_loc('ARRIVALDATE')]) - 
                 IMO_info.iloc[len(IMO_info)-1, IMO_info.columns.get_loc('SAILDATE')])/np.timedelta64(1,'D'))/30.44),0)

    period_used_IMO.loc[period_used_IMO.index[period_used_IMO['IMO'] == each_IMO].item(), 'period_used'] = mont_used
# create condition to asign categories for each period used
conditions = [period_used_IMO['period_used'] <= 3,
    (period_used_IMO['period_used'] >= 4) & (period_used_IMO['period_used'] <= 6),
    (period_used_IMO['period_used'] >= 7) & (period_used_IMO['period_used'] <= 9),
    period_used_IMO['period_used'] >= 10]
values = [
    'One quater', 'Two quaters', 'Three quarters','Full year'
]
# create a data table with assigned categories
period_used_IMO['range'] = np.select(conditions, values, default=0)
period_used_IMO = period_used_IMO.drop(['period_used'], axis =1)
# plot distribution of ships used over certain period
gr_period_used_IMO = period_used_IMO.groupby('range').count()
gr_period_used_IMO.plot.bar()
# stacked bar showing ship types used in a certain period
period_used_IMOw_Stype_df = period_used_IMO.merge(
    IMO_w_portcall_freq_uniq_df[['SHIPTYPE', 'IMO']],
    left_on='IMO',
    right_on='IMO',
    how='left'
)
stacked_bar_shiptype = period_used_IMOw_Stype_df.groupby(['range','SHIPTYPE']).size().unstack()
stacked_bar_shiptype = period_used_IMOw_Stype_df.groupby('range')['SHIPTYPE'].value_counts().unstack('SHIPTYPE')
ax = stacked_bar_shiptype.plot.bar(stacked=True)
plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
plt.title('Total active time of vessels')
ax.set_xlabel('Time on sea')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=0)
stacked_bar_shiptype = period_used_IMOw_Stype_df.groupby('range')['SHIPTYPE'].value_counts(normalize =  True).unstack('SHIPTYPE')
stacked_bar_shiptype.plot.bar(stacked=True)
plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
plt.title('Total active time of vessels')
ax.set_xlabel('Time on sea')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=0)
# visualize the relationship between port frequency, vessels visit them, and duration time

# Group by port and vessel, calculate average duration
pivot_df = tanker_were_to_NL_ww.groupby(['PORTNAME', 'SHIPTYPE'])['DURATIONINPORT_HR'].mean().unstack()

# Keep only top 10 ports by visit count
top_ports = tanker_were_to_NL_ww['PORTNAME'].value_counts().head(10).index
pivot_df = pivot_df.loc[top_ports]

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, cmap='YlGnBu')
plt.title("Avg Stay Duration by Vessel and Port")
plt.xlabel("Vessel IMO")
plt.ylabel("Port")
plt.tight_layout()
plt.show()

# Visit frequency per port and ship type
freq = tanker_were_to_NL_ww.groupby(['PORTNAME', 'SHIPTYPE']).size().unstack(fill_value=0)

# Get top 10 ports
top_ports = tanker_were_to_NL_ww['PORTNAME'].value_counts().head(20).index
freq = freq.loc[top_ports]

plt.figure(figsize=(12, 6))
sns.heatmap(freq, fmt="d", cmap="Blues")
plt.title("Visit Frequency by Vessel Type and Top 10 Ports")
plt.xlabel("Vessel Type")
plt.ylabel("Port")
plt.tight_layout()
plt.show()

# IMO
# Determine outliners IMO that have a fewer port calls
IMO_w_portcall_freq_uniq_df
    # determine the relationship between tanker types and hour in port
ax_freq_type = sns.boxplot(data = IMO_w_portcall_freq_uniq_df[IMO_w_portcall_freq_uniq_df['SHIPTYPE'] == 'Crude Oil Tanker'], x = 'SHIPTYPE', y = 'Frequency')  

ax_freq_type.tick_params(axis='x', rotation= 90)
plt.title('Relationship between tanker types and their port call frequency')
plt.show()
# UNLOCO
portname, countport = np.unique(tanker_were_to_NL_ww['PORTNAME'], return_counts = True)
len(portname)

harbor_names = pd.read_csv("./IVS3_CodeTabel_Plaatsen.csv")
extra_harbor_names = pd.read_csv('./port_unloco.csv')
port_names = pd.concat([harbor_names, extra_harbor_names], ignore_index=True, sort=False)

columns_to_keep = ['Code', 'Plaats']
harbor_names = port_names[columns_to_keep]
port_names.loc[:,'Plaats'] = port_names['Plaats'].str.replace(r'\s*\(.*?\)', '', regex=True)

tanker_were_to_NL_ww['PORTNAME'] = tanker_were_to_NL_ww['PORTNAME'].str.upper()
    # Merge (geo)dataframes
tanker_were_to_NL_ww_with_codes = tanker_were_to_NL_ww.merge(port_names,
        left_on='PORTNAME',
        right_on='Plaats',
        how='left'
    )

tanker_were_to_NL_ww_with_codes['Code'].isnull().sum()
tanker_were_to_NL_ww_with_codes_copy = tanker_were_to_NL_ww_with_codes[['PORTNAME', 'Code']]
tanker_were_to_NL_ww_with_codes_copy = tanker_were_to_NL_ww_with_codes_copy.drop_duplicates()
tanker_were_to_NL_ww_with_codes_copy = tanker_were_to_NL_ww_with_codes_copy[tanker_were_to_NL_ww_with_codes_copy.isna().any(axis=1)]

print(tanker_were_to_NL_ww_with_codes_copy['PORTNAME'])
fre_port = tanker_were_to_NL_ww_with_codes['PORTNAME'].value_counts()
ctry_fre = tanker_were_to_NL_ww['COUNTRY '].value_counts().to_frame(name = "FREQ")






# %% explore data for tankers were to RU (ww)
# check for NaN
tanker_were_to_RU_ww = pp.check_nan_duplicate(tanker_were_to_RU_ww)


# check number of IMO containing both bunkering tanker (oil) and , (inland waterway)
tanker_were_to_RU_ww['SHIPTYPE'].unique()

# Check for the standard form of IMO
pp.check_IMO(tanker_were_to_RU_ww)
# validate the order consistancy of arrival date, sail date, last port call sail date
pp.validate_timestamp_sign(tanker_were_to_RU_ww)

# check identical arrival date, sail date, last POC sail date

arrvl_sail_dup_RU, IMO_freq_arrvl_sail_dup_RU, arrvl_sail_lastPOC_dup_RU, IMO_freq_arrvl_sail_lastPOC_dup_RU = pp.check_duplicated_timestamp(tanker_were_to_RU_ww)
tanker_were_to_RU_ww_9885879 = tanker_were_to_RU_ww[tanker_were_to_RU_ww['IMO'] == 9885879]

# Calculate dublicated previous sail data and the next arrival date for each ship-or too small time gap for traveling
count_small_time_gap_0_RU = 0
count_small_time_gap_smaller1hr_RU = 0
for IMO_nr in tanker_were_to_RU_ww['IMO'].unique():
    tanker_nr = tanker_were_to_RU_ww[tanker_were_to_RU_ww['IMO'] == IMO_nr]
    tanker_nr['TIME_TRAVEL'] = tanker_nr['ARRIVALDATE'] - tanker_nr['SAILDATE'].shift(1)
    tanker_nr['TIME_TRAVEL_SEC'] = (tanker_nr['TIME_TRAVEL'].dt.days * seconds_in_day + tanker_nr['TIME_TRAVEL'].dt.seconds)
    # number_small_time_gap = tanker_nr['TIME_TRAVEL_SEC'][tanker_nr['TIME_TRAVEL_SEC'] < 600]
    # count_small_time_gap = count_small_time_gap + len(number_small_time_gap)
    too_small_time_gap_index = tanker_nr.index[tanker_nr['TIME_TRAVEL_SEC'] == 0].tolist()
    count_small_time_gap_0_RU = count_small_time_gap_0_RU + len(too_small_time_gap_index)
    too_small_time_gap_index_smaller1hr = tanker_nr.index[tanker_nr['TIME_TRAVEL_SEC'] < 3600].tolist()
    count_small_time_gap_smaller1hr_RU = count_small_time_gap_smaller1hr_RU + len(too_small_time_gap_index_smaller1hr)
    print("IMO",IMO_nr,"has this list of IMO containing too small travel time gap:", too_small_time_gap_index)
    

# standadize port name
tanker_were_to_RU_ww['PORTNAME'] = tanker_were_to_RU_ww['PORTNAME'].map(lambda x: pp.standardize_port_name(x))

# standadize ship name
tanker_were_to_RU_ww['SHIPTYPE'] = tanker_were_to_RU_ww['SHIPTYPE'].map(lambda x: pp.standadize_ship_type(x))

# Number of IMO Vessels by Frequency of Port Calls
# Frequency of port call per IMO
IMO_w_portcall_freq_uniq_df_RU = pp.IMO_freq_port_call(tanker_were_to_RU_ww).drop_duplicates()
# Contribution of ship types to the total frequency of each bin of port call frequency
pp.plot_NO_ships_per_freq(IMO_w_portcall_freq_uniq_df_RU)

# HOURS IN PORT
# validate hour in port
tanker_were_to_RU_ww,large_diff_HRinPrt_actHr_list_RU = pp.validate_hrinport(tanker_were_to_RU_ww)
# plot Hours in Port Frequency
pp.plot_hourinport_freq_for_IMO(tanker_were_to_RU_ww, 'Hours in Port Frequency (tankers were to RU-ww)')
tanker_were_to_RU_ww['DURATIONINPORT_HR'].quantile(q = 0.75)
tanker_were_to_RU_ww['DURATIONINPORT_HR'].quantile(q = 0.90)

# plot the number of IMO in each ship type
stacked_bar_shiptype_RU = IMO_w_portcall_freq_uniq_df_RU.groupby('SHIPTYPE').size().reset_index()
stacked_bar_shiptype_RU.rename(columns = {0 : 'Nr.IMO'}, inplace = True)
stacked_bar_shiptype_RU.rename( columns={0 :'Nr.IMO'}, inplace=True )
ax = stacked_bar_shiptype_RU.plot.bar('SHIPTYPE', 'Nr.IMO')
ax.set_xlabel('SHIP TYPE')
ax.set_ylabel('Nr. IMO')
plt.title("Total number of IMO in each ship type")
# determine the relationship between tanker types and hour in port
# determine the relationship between tanker types and the number of port frequency
pp.plot_boxplot(tanker_were_to_RU_ww.sort_values(by = 'SHIPTYPE'),IMO_w_portcall_freq_uniq_df_RU.sort_values(by = 'SHIPTYPE'))


# determine outliners of hours in port 
# for ship_type in tanker_were_to_NL_ww['SHIPTYPE'].unique().tolist():
#     #tanker_were_to_NL_ww
#     shiptype = tanker_were_to_NL_ww[tanker_were_to_NL_ww['SHIPTYPE'] == ship_type]
#     std_shiptype_hrinport = stats.stdev(shiptype['DURATIONINPORT_HR'])
#     mean_shiptype_hrinport = shiptype['DURATIONINPORT_HR'].mean(axis = 0)
#     lower_bound = mean_shiptype_hrinport - 2 * std_shiptype_hrinport
#     upper_bound = mean_shiptype_hrinport + 2 * std_shiptype_hrinport
#     plt.figure(figsize = (12,6))
#     plt.scatter(x = shiptype['ARRIVALDATE'], y = shiptype['DURATIONINPORT_HR'], label = 'Hours in port', color = 'blue', alpha = 0.6)
#     plt.axhline(lower_bound, color = 'red', linestyle = '--', label = '-2 STD')
#     plt.axhline(upper_bound, color = 'red', linestyle = '--', label = '+2 STD')
#     plt.axhline(mean_shiptype_hrinport, color =  'green', linestyle = '--', label = 'mean')
#     plt.fill_between(shiptype['ARRIVALDATE'], lower_bound, upper_bound, color = 'gray', alpha = 0.2, label= '±2 STD Region')
#     plt.title(f'Outliners of hours in port for {ship_type}', fontsize = 30)
#     plt.gca().set_xlabel('Arrival time', fontsize=20)
#     plt.gca().set_ylabel('Hours in port', fontsize=20)
#     plt.xticks(fontsize = 15)
#     plt.yticks(fontsize = 15)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# determine outliners of hours in port 
# for ship_type in IMO_w_portcall_freq_uniq_df['SHIPTYPE'].unique().tolist():
#     #tanker_were_to_NL_ww
#     shiptype = IMO_w_portcall_freq_uniq_df[IMO_w_portcall_freq_uniq_df['SHIPTYPE'] == ship_type]
#     if len(shiptype) <2:
#         print(shiptype['SHIPTYPE'], 'has number of IMO smaller than 2')
#         continue
#     std_shiptype_hrinport = stats.stdev(shiptype['Frequency'])
#     mean_shiptype_hrinport = shiptype['Frequency'].mean(axis = 0)
#     median_shiptype_hrinport = shiptype['Frequency'].median(axis = 0)
#     lower_bound = shiptype['Frequency'].quantile(q=0.25) - 1.5*std_shiptype_hrinport

#     plt.figure(figsize = (12,6))
#     plt.scatter(x = shiptype['IMO'], y = shiptype['Frequency'], label = 'Total number of call', color = 'blue', alpha = 0.6)
#     plt.axhline(lower_bound, color = 'red', linestyle = '--', label = '-1.5 STD')

#     plt.axhline(mean_shiptype_hrinport, color =  'green', linestyle = '--', label = 'mean')
#     plt.fill_between(shiptype['IMO'], lower_bound, mean_shiptype_hrinport, color = 'gray', alpha = 0.2, label= '-1.5 STD Region')
#     plt.title(f'Outliners of frequency of port calls for {ship_type}', fontsize = 30)
#     plt.gca().set_xlabel('IMO', fontsize=20)
#     plt.gca().set_ylabel('Total number of call', fontsize=20)
#     plt.xticks(fontsize = 15)
#     plt.yticks(fontsize = 15)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    
# Amount of IMO full used, part time used


period_used_IMO_RU = pd.DataFrame(tanker_were_to_RU_ww['IMO'].unique(), columns = ['IMO'])

period_used_IMO_RU['period_used'] = 'Nan'
period_used_IMO_RU['range'] = 'Nan'
# calculate number of months a IMO were on sea
for each_IMO in tanker_were_to_RU_ww['IMO'].unique():
    IMO_info = tanker_were_to_RU_ww[tanker_were_to_RU_ww['IMO'] == each_IMO]
    mont_used = round(abs((((IMO_info.iloc[0, IMO_info.columns.get_loc('ARRIVALDATE')]) - 
                 IMO_info.iloc[len(IMO_info)-1, IMO_info.columns.get_loc('SAILDATE')])/np.timedelta64(1,'D'))/30.44),0)

    period_used_IMO_RU.loc[period_used_IMO_RU.index[period_used_IMO_RU['IMO'] == each_IMO].item(), 'period_used'] = mont_used
# create condition to asign categories for each period used
conditions_RU = [period_used_IMO_RU['period_used'] <= 3,
    (period_used_IMO_RU['period_used'] >= 4) & (period_used_IMO_RU['period_used'] <= 6),
    (period_used_IMO_RU['period_used'] >= 7) & (period_used_IMO_RU['period_used'] <= 9),
    period_used_IMO_RU['period_used'] >= 10]
values = [
    'One quater', 'Two quaters', 'Three quarters','Full year'
]
# create a data table with assigned categories
period_used_IMO_RU['range'] = np.select(conditions_RU, values, default=0)
period_used_IMO_RU = period_used_IMO_RU.drop(['period_used'], axis =1)
# plot distribution of ships used over certain period
gr_period_used_IMO_RU = period_used_IMO_RU.groupby('range').count()
gr_period_used_IMO_RU.plot.bar()
# stacked bar showing ship types used in a certain period
period_used_IMOw_Stype_df_RU = period_used_IMO_RU.merge(
    IMO_w_portcall_freq_uniq_df_RU[['SHIPTYPE', 'IMO']],
    left_on='IMO',
    right_on='IMO',
    how='left'
)
stacked_bar_shiptype_RU = period_used_IMOw_Stype_df_RU.groupby(['range','SHIPTYPE']).size().unstack()
stacked_bar_shiptype_RU = period_used_IMOw_Stype_df_RU.groupby('range')['SHIPTYPE'].value_counts().unstack('SHIPTYPE')
ax = stacked_bar_shiptype_RU.plot.bar(stacked=True)
plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
plt.title('Total active time of vessels')
ax.set_xlabel('Time on sea')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=0)

# visualize the relationship between port frequency, vessels visit them, and duration time

# Group by port and vessel, calculate average duration
pivot_df_RU = tanker_were_to_RU_ww.groupby(['PORTNAME', 'SHIPTYPE'])['DURATIONINPORT_HR'].mean().unstack()

# Keep only top 10 ports by visit count
top_ports_RU = tanker_were_to_RU_ww['PORTNAME'].value_counts().head(10).index
pivot_df_RU = pivot_df_RU.loc[top_ports_RU]

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df_RU, cmap='YlGnBu')
plt.title("Avg Stay Duration by Vessel and Port")
plt.xlabel("Vessel IMO")
plt.ylabel("Port")
plt.tight_layout()
plt.show()

# Visit frequency per port and ship type
freq_RU = tanker_were_to_RU_ww.groupby(['PORTNAME', 'SHIPTYPE']).size().unstack(fill_value=0)

# Get top 10 ports
top_ports_RU = tanker_were_to_RU_ww['PORTNAME'].value_counts().head(20).index
freq_RU = freq_RU.loc[top_ports_RU]

plt.figure(figsize=(12, 6))
sns.heatmap(freq_RU, fmt="d", cmap="Blues")
plt.title("Visit Frequency by Vessel Type and Top 10 Ports")
plt.xlabel("Vessel Type")
plt.ylabel("Port")
plt.tight_layout()
plt.show()

# Pie chart the contribution of countries


tanker_were_to_RU_ww.columns
# Get unique ship types
ship_types = tanker_were_to_RU_ww['SHIPTYPE'].unique()

# Loop over each ship type to create pie charts
for ship_type in ship_types:
    # Filter data for current ship type
    df_sub = tanker_were_to_RU_ww[tanker_were_to_RU_ww['SHIPTYPE'] == ship_type]
    
    # Count number of visits per country
    country_counts = df_sub['COUNTRY '].value_counts()
    
    # Plot pie chart
    plt.figure(figsize=(6, 6))
    country_counts.plot.pie(
        autopct='%1.1f%%',
        startangle=140,
        counterclock=False,
        shadow=True
    )
    plt.title(f"{ship_type}", fontsize=20)
    plt.ylabel('')  # Remove y-label for cleaner look
    plt.tight_layout()
    plt.show()
    
    
tanker_were_to_RU_ww.columns
# Get unique ship types
ship_types = tanker_were_to_NL_ww['SHIPTYPE'].unique()

# Loop over each ship type to create pie charts
for ship_type in ship_types:
    # Filter data for current ship type
    df_sub_RU = tanker_were_to_NL_ww[tanker_were_to_NL_ww['SHIPTYPE'] == ship_type]
    
    # Count number of visits per country
    country_counts = df_sub_RU['COUNTRY '].value_counts()
    
    # Plot pie chart
    plt.figure(figsize=(6, 6))
    country_counts.plot.pie(
        autopct='%1.1f%%',
        startangle=140,
        counterclock=False,
        shadow=True
    )
    plt.title(f"{ship_type}", fontsize=20)
    plt.ylabel('')  # Remove y-label for cleaner look
    plt.tight_layout()
    plt.show()