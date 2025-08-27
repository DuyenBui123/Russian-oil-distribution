# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 07:55:49 2025

@author: Duyen
"""

import os
cwd = os.getcwd()
os.chdir(cwd)
import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from Code import data_processing as pr
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# %% import data

# alltankers_adjusted = pd.read_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv',
#                                   dtype= {'IMO' : 'int64', 'DepPort':'object',
#                                           'ArrPort':'object',
#                                           'ShipType':'object',
#                                           'Country':'object',
#                                           'Arr_Country':'object'}, 
#                                   parse_dates= ['DepDate', 'ArrDate'],
#                                   index_col = 0).rename_axis('Index')
alltankers_adjusted = pd.read_csv('./RU_oil_tankers_data.csv',
                                  dtype= {'IMO' : 'int64', 'DepPort':'object',
                                          'ArrPort':'object',
                                          'ShipType':'object',
                                          'Country':'object',
                                          'Arr_Country':'object'}, 
                                  parse_dates= ['DepDate', 'ArrDate'],
                                  index_col = 0).rename_axis('Index')
# correcting data type
for col in ['TravelTime', 'BerthTime']:
    
    alltankers_adjusted[col] = pd.to_timedelta(alltankers_adjusted[col])

country_of_interest = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Kazakhstan', 'Malaysia']


route = joblib.load('./processing/pr_inter_output/potential_routes_loop_nrRU_1_time1m_nrtotport5.joblib')
# route = joblib.load('./wloop_potential_routes_nrRU_1_nrtotport4.joblib')
countries = alltankers_adjusted['Country'].unique()

lookup = {
    'Net exporter': {"Algeria", "Angola", "Azerbaijan", "Bahrain", "Brunei", "Colombia",
    "Congo (Democratic Republic)", "Congo (Republic)", "Ecuador",
    "Equatorial Guinea", "Falkland Islands", "Gabon", "Guyana", "Iran",
    "Iraq", "Kazakhstan", "Kuwait", "Libya", "Mauritania", "Nigeria",
    "Norway", "Oman", "Qatar", "Saudi Arabia", "Sudan",
    "Trinidad & Tobago", "Turkmenistan", "Venezuela", "Yemen"},
    'Mixed': {"United States of America",
    "United Kingdom",
    "Brazil",
    "Malaysia",
    "Indonesia",
    "Canada",
    "Mexico",
    "Russia",
    "China",
    "India",
    "United Arab Emirates",
    "Singapore",
    "Netherlands",
    "Italy",
    "Turkey",
    "South Africa"},
    'Net importer': {"Argentina", "Australia", "Bangladesh", "Belgium", "Benin", "Bulgaria", 
    "Cameroon", "Canary Islands", "Cape Verde Islands", "Chile", "Costa Rica", 
    "Cote d'Ivoire", "Croatia", "Cuba", "Curacao", "Cyprus", "Denmark", 
    "Dominican Republic", "Egypt", "El Salvador", "Estonia", "Faeroe Islands", 
    "Finland", "France", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", 
    "Guam", "Guatemala", "Guinea", "Hong Kong", "Ireland", "Israel", "Jamaica", 
    "Japan", "Kenya", "Latvia", "Liberia", "Lithuania", "Madeira", "Malta", 
    "Martinique", "Morocco", "Myanmar", "Namibia", "New Caledonia", "New Zealand", 
    "Nicaragua", "Pakistan", "Panama", "Papua New Guinea", "Peru", "Philippines", 
    "Poland", "Portugal", "Puerto Rico", "Romania", "Senegal", "Sierra Leone", 
    "Sint Eustatius", "Slovenia", "Solomon Islands", "Spain", "Sri Lanka", 
    "St Helena Island", "St Lucia", "Sweden", "Tanzania", "Thailand", "Tunisia", 
    "Ukraine", "Uruguay", "Vietnam", "Virgin Islands (US)", "Western Sahara", "Korea (South)", "Chinese Taipei (Taiwan)", "Bahamas", "Jordan", "Aruba",
    "Guernsey", "Jersey", "Lebanon", "Albania", "Georgia", "Djibouti",
    "Mozambique", "Honduras", "Togo", "Korea (North)"}
}

len(countries)

imo_w_stat = pd.DataFrame({
    'Country': countries,
    'Status': [next(status for status, group in lookup.items() if country in group) for country in countries]
})



# %% OLD CODE
# # TOTAL 3 PORTs
# route_RU_int_NL_matched_imoNr,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
#     route, alltankers_adjusted, 5, 3,  False, oiltype = 'all', loop_type = 'country')
# route_dir = []
# # check how many extracted route matched with the reported suspecious routes
# for df in route_RU_int_NL_matched_imoNr:
#     if df.iloc[0, df.columns.get_loc('Arr_Country')] in (country_of_interest):
#                route_dir.append(df)
# perc_route_matched_reported = (len(route_dir)/len(route_RU_int_NL_matched_imoNr))*100
# # extract and analyze the IMO per row or per trip
# per_row = pd.DataFrame(columns = alltankers_adjusted.columns)
# for df in route_dir:
#     first_row = df[0:1]
#     per_row = pd.concat([per_row, first_row])
    
# # drop dubplicated
# next_trip = pd.DataFrame(columns = alltankers_adjusted.columns)
# per_row = per_row.drop_duplicates()
# for row in range(len(per_row)):
#     imo = per_row.iloc[row]['IMO']
#     imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
#     row_index = per_row.iloc[row:row+1].index[0] 
#     for nr_row in range(len(imo_poc)):
#         if row_index == imo_poc.iloc[nr_row:nr_row+1].index[0]:
#             next_trip = pd.concat([next_trip, imo_poc.iloc[nr_row+1:nr_row+2]])
#             continue
        
# # # check if there is any same set of imo occurs

# country_pair = tuple(zip(next_trip['Country'], next_trip['Arr_Country']))
# counter_cntr_pair = collections.Counter(country_pair)
# counter_ctry_arr = collections.Counter(next_trip['Arr_Country'])


# # TOTAL 4 PORTS
# # check the next stop of these IMO

# max_size = max(len(inner) for inner in route)
# perc_route_matched_diffports = []
# counter_ctry_collect = []
# counter_ctry_arr_collect = []
# hotspot_toward_collect = []

# counter_port_pair_collect = []
# for n in range(3, max_size+1):
#     nr_it_per_nr_port = 0
#     count_ctry_pair = []
#     count_ctry_arr = []
#     count_port_dep = []
#     hotspot_toward = []
#     for m in range(2, max_size):

        
#         if m < n:
#             nr_it_per_nr_port = nr_it_per_nr_port +1

#             route_RU_int_NL_matched_imoNr2,trip_freq_dict2, country_seq2, port_sequence2 = pr.route_seq_matched_nrimo_par(
#                 route, alltankers_adjusted, n, m,  False, oiltype = 'all', loop_type = 'country')
#             route_dir2 = []
#             # check how many extracted route matched with the reported suspecious routes
#             for df in route_RU_int_NL_matched_imoNr2:
#                 if df.iloc[0, df.columns.get_loc('Arr_Country')] in (country_of_interest):
#                            route_dir2.append(df)
#             perc_route_matched_reported2 = (len(route_dir2)/len(route_RU_int_NL_matched_imoNr2))*100   
#             perc_route_matched_diffports.append(perc_route_matched_reported2)
            
    
#             # extract and analyze the first IMO from RU per trip
#             per_row2 = pd.DataFrame(columns = alltankers_adjusted.columns)
#             for df in route_dir2:
#                 first_row = df[0:1]
#                 per_row2 = pd.concat([per_row2, first_row])
                
#             # drop dubplicated
#             # check what is the next trip of the IMO
#             next_trip2 = pd.DataFrame(columns = alltankers_adjusted.columns)
#             per_row2 = per_row2.drop_duplicates()
#             for row in range(len(per_row2)):
#                 imo = per_row2.iloc[row]['IMO']
#                 imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
#                 row_index = per_row2.iloc[row:row+1].index[0] 
#                 for nr_row in range(len(imo_poc)):
#                     if row_index == imo_poc.iloc[nr_row:nr_row+1].index[0]:
#                         next_trip2 = pd.concat([next_trip2, imo_poc.iloc[nr_row+1:nr_row+2]])
#                         continue
            
            
#             country_pair2 = tuple(zip(next_trip2['Country'], next_trip2['Arr_Country']))
#             counter_cntr_pair2 = collections.Counter(country_pair2)
#             count_ctry_pair.append(counter_cntr_pair2)
#             counter_ctry_arr2 = collections.Counter(next_trip2['Arr_Country'])
#             count_ctry_arr.append(counter_ctry_arr2)
    
#             counter_port_pair = collections.Counter(next_trip2['DepPort'])
#             count_port_dep.append(counter_port_pair)
#             # select trips have two different consecutive IMO at the first share ports of the routes

#             for df in route_dir2:
#                 if df['IMO'].iloc[0] != df['IMO'].iloc[1]:
#                     hotspot_toward.append(df)
#     counter_ctry_collect.append(count_ctry_pair)
#     counter_ctry_arr_collect.append(count_ctry_arr)
#     counter_port_pair_collect.append(count_port_dep)
#     hotspot_toward_collect.append(hotspot_toward)

# pr.depth(counter_ctry_collect)
# port3 = counter_port_pair_collect[0][0]

# port4_1 = counter_port_pair_collect[1][0]
# port4_2 = counter_port_pair_collect[1][1]
# port4 = collections.Counter()
# for key in set(port4_1) | set(port4_2):  # union of keys
#     port4[key] = round((port4_1.get(key, 0) + port4_2.get(key, 0)) / len(counter_port_pair_collect[1]), 0)           

# port5_1 = counter_port_pair_collect[2][0]
# port5_2 = counter_port_pair_collect[2][1]
# port5_3 = counter_port_pair_collect[2][2]
# port5 = collections.Counter()
# for key in set(port5_1) | set(port5_2) | set(port5_3):  # union of keys
#     port5[key] = round((port5_1.get(key, 0) + port5_2.get(key, 0) + port5_3.get(key, 0)) / len(counter_port_pair_collect[2]), 0)
# # Select top 5 based on any of them (union, then take top 5 of combined)
# combined_keys = set(port3.keys()) | set(port4.keys()) | set(port5.keys())
# top_keys = sorted(combined_keys, key=lambda k: max(port3.get(k,0), port4.get(k,0), port5.get(k,0)), reverse=True)[:5]

# # Data for plotting
# port3_vals = [port3.get(k, 0) for k in top_keys]
# port4_vals = [port4.get(k, 0) for k in top_keys]
# port5_vals = [port5.get(k, 0) for k in top_keys]

# x = np.arange(len(top_keys))  # label locations
# width = 0.25  # bar width

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(x - width, port3_vals, width, label='Port 3')
# ax.bar(x, port4_vals, width, label='Port 4')
# ax.bar(x + width, port5_vals, width, label='Port 5')

# # Labels and title with increased font sizes
# ax.set_ylabel('Values', fontsize=14)
# ax.set_title('Top 5 Ports Comparison', fontsize=16)
# ax.set_xticks(x)
# ax.set_xticklabels(top_keys, rotation=45, fontsize=20)
# ax.tick_params(axis='y', labelsize=20)
# ax.legend(fontsize=20)

# plt.tight_layout()
# plt.show()



# max_size = max(len(inner) for inner in route)

# %% focus on two IMO in the entire routes
max_size = max(len(inner) for inner in route)
hotspot_toward_collect = []

for n in range(3, max_size+1):


            route_RU_int_NL_matched_imoNr2,trip_freq_dict2, country_seq2, port_sequence2 = pr.route_seq_matched_nrimo_par(
                route, alltankers_adjusted, n, 2,  False, oiltype = 'all', loop_type = 'country')
            route_dir2 = []
            # check how many extracted route matched with the reported suspecious routes
            for df in route_RU_int_NL_matched_imoNr2:
                if df.iloc[0, df.columns.get_loc('Arr_Country')] in (country_of_interest):
                           route_dir2.append(df)
            hotspot_toward = []
            

            for df in route_dir2:
                if df['IMO'].iloc[0] != df['IMO'].iloc[1]:
                    hotspot_toward.append(df)
            hotspot_toward_collect.append(hotspot_toward)

# pattern from RU to hotspot
count_ctry_pair = []
count_ctry_arr = []
count_port_dep = []
for route_dir2 in hotspot_toward_collect:
    # extract and analyze the first IMO from RU per trip
    per_row2 = pd.DataFrame(columns = alltankers_adjusted.columns)
    for df in route_dir2:
        first_row = df[0:1]
        per_row2 = pd.concat([per_row2, first_row])
        
    # drop dubplicated
    # check what is the next trip of the IMO
    next_trip2 = pd.DataFrame(columns = alltankers_adjusted.columns)
    per_row2 = per_row2.drop_duplicates()
    for row in range(len(per_row2)):
        imo = per_row2.iloc[row]['IMO']
        imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
        row_index = per_row2.iloc[row:row+1].index[0] 
        for nr_row in range(len(imo_poc)):
            if row_index == imo_poc.iloc[nr_row:nr_row+1].index[0]:
                next_trip2 = pd.concat([next_trip2, imo_poc.iloc[nr_row+1:nr_row+2]])
                continue


    country_pair2 = tuple(zip(next_trip2['Country'], next_trip2['Arr_Country']))
    counter_cntr_pair2 = collections.Counter(country_pair2)
    count_ctry_pair.append(counter_cntr_pair2)
    counter_ctry_arr2 = collections.Counter(next_trip2['Arr_Country'])
    count_ctry_arr.append(counter_ctry_arr2)
    
    counter_port_pair = collections.Counter(next_trip2['DepPort'])
    count_port_dep.append(counter_port_pair)

# Plot

port3 = count_port_dep[0]
port4 = count_port_dep[1]
port5 = count_port_dep[2]

# Select top 5 based on any of them (union, then take top 5 of combined)
combined_keys = set(port3.keys()) | set(port4.keys()) | set(port5.keys())
top_keys = sorted(combined_keys, key=lambda k: max(port3.get(k,0), port4.get(k,0), port5.get(k,0)), reverse=True)[:5]

# Data for plotting
port3_vals = [port3.get(k, 0) for k in top_keys]
port4_vals = [port4.get(k, 0) for k in top_keys]
port5_vals = [port5.get(k, 0) for k in top_keys]

x = np.arange(len(top_keys))  # label locations
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, port3_vals, width, label='3 Ports')
ax.bar(x, port4_vals, width, label='4 Ports')
ax.bar(x + width, port5_vals, width, label='5 Ports')

# Labels and title with increased font sizes
ax.set_ylabel('Frequency', fontsize=20)
ax.set_title('Top 5 Ports for different port depths', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(top_keys, rotation=45, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()


# Step 1: Process counters into top5 + Others
final_results = []
for counter in count_ctry_arr:
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_items[:5]
    others_sum = sum(v for _, v in sorted_items[5:])
    top5.append(("Others", others_sum))
    final_results.append(collections.Counter(dict(top5)))

# Step 2: Combine all keys from all results
all_keys = sorted(set().union(*final_results))

# Step 3: Extract aligned values for plotting
values = [[res.get(k, 0) for k in all_keys] for res in final_results]

# Step 4: Plot
x = np.arange(len(all_keys))
width = 0.25  # Adjusted so 3 bars fit side-by-side

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['skyblue', 'orange', 'black']
labels = ['3 ports', '4 ports', '5 ports']

for i, (vals, label, color) in enumerate(zip(values, labels, colors)):
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)

# Formatting
ax.set_ylabel('Counts', fontsize=20)
ax.set_title('Top 5 return countries from a hotspot', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(all_keys, rotation=45, ha='right', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()
####################################
# select trips have two different consecutive IMO at the first share ports of the routes

snd_trip = pd.DataFrame(columns = alltankers_adjusted.columns)

for df in hotspot_toward_collect[1]:
    second_trip = df.iloc[1:2]
    snd_trip = pd.concat([snd_trip, second_trip])
    
snd_trip = snd_trip.drop_duplicates()
country_pair4 = tuple(zip(snd_trip['Country'], snd_trip['Arr_Country']))
counter_cntr_pair4 = collections.Counter(country_pair4)
counter_ctry_arr4 = collections.Counter(snd_trip['Arr_Country'])    
counter_port_dep4 = collections.Counter(snd_trip['ArrPort']) 



snd_trip = pd.DataFrame(columns = alltankers_adjusted.columns)

for df in hotspot_toward_collect[2]:
    second_trip = df.iloc[1:2]
    snd_trip = pd.concat([snd_trip, second_trip])
    
snd_trip = snd_trip.drop_duplicates()
country_pair5_1 = tuple(zip(snd_trip['Country'], snd_trip['Arr_Country']))
counter_cntr_pair5_1 = collections.Counter(country_pair5_1)
counter_ctry_arr5_1 = collections.Counter(snd_trip['Arr_Country'])    
counter_port_dep5_1 = collections.Counter(snd_trip['ArrPort']) 

snd_trip = pd.DataFrame(columns = alltankers_adjusted.columns)

for df in hotspot_toward_collect[2]:
    second_trip = df.iloc[2:3]
    snd_trip = pd.concat([snd_trip, second_trip])
    
snd_trip = snd_trip.drop_duplicates()
country_pair5_2 = tuple(zip(snd_trip['Country'], snd_trip['Arr_Country']))
counter_cntr_pair5_2 = collections.Counter(country_pair5_2)
counter_ctry_arr5_2 = collections.Counter(snd_trip['Arr_Country'])    
counter_port_dep5_2 = collections.Counter(snd_trip['ArrPort']) 

counter_port_dep5 = counter_port_dep5_1 + counter_port_dep5_1
counter_ctry_arr5 = counter_ctry_arr5_1 + counter_ctry_arr5_2


# Put your counters in a list so we can process in a loop
counters = [counter_ctry_arr4, counter_ctry_arr5_1, counter_ctry_arr5_2]
labels = ['seg 2 of 4 ports', 'seg 2 of 5 ports', 'seg 3 of 5 ports']
colors = ['skyblue', 'orange', 'black']

# Step 1: Convert each counter to Top 5 + Others
final_results = []
for counter in counters:
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_items[:5]
    others_sum = sum(v for _, v in sorted_items[5:])
    top5.append(("Others", others_sum))
    final_results.append(collections.Counter(dict(top5)))

# Step 2: Get all keys across all counters
all_keys = sorted(set().union(*final_results))

# Step 3: Prepare aligned values
values = [[res.get(k, 0) for k in all_keys] for res in final_results]

# Step 4: Plot
x = np.arange(len(all_keys))
width = 0.25  # Adjust bar width for 3 datasets

fig, ax = plt.subplots(figsize=(12, 6))
for i, (vals, label, color) in enumerate(zip(values, labels, colors)):
    ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)

# Formatting
ax.set_ylabel('Counts', fontsize= 20)
ax.set_title('Top 5 countries an IMO stops by before reaching NL', fontsize=25)
ax.set_xticks(x)
ax.set_xticklabels(all_keys, rotation=45, ha='right', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()


# %% check pattern of IMO arrive to NL

beyond_lasttrip = []


for lst in range(len(hotspot_toward_collect)):
    route = hotspot_toward_collect[lst]
    per_row2 = pd.DataFrame(columns = alltankers_adjusted.columns)
    for df in route:


            last_row = df.iloc[[-1]]
            per_row2 = pd.concat([per_row2, last_row])
    per_row2 = per_row2.drop_duplicates()
    next_trip2 = pd.DataFrame(columns = alltankers_adjusted.columns)
    for row in range(len(per_row2)):
        imo = per_row2.iloc[row]['IMO']
        imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
        row_index = per_row2.iloc[row:row+1].index[0] 
        for nr_row in range(len(imo_poc)):
            if row_index == imo_poc.iloc[nr_row:nr_row+1].index[0]:
                next_trip2 = pd.concat([next_trip2, imo_poc.iloc[nr_row+1:nr_row+2]])
                continue
    beyond_lasttrip.append(next_trip2)
                
            # drop dubplicated
            # check what is the next trip of the IMO
            

from_NL_to_port =  collections.Counter(beyond_lasttrip[0]['ArrPort'])
from_NL_to_cntry =  collections.Counter(beyond_lasttrip[0]['Arr_Country']) 
from_NL_to_port2 =  collections.Counter(beyond_lasttrip[1]['ArrPort'])
from_NL_to_cntry2 =  collections.Counter(beyond_lasttrip[1]['Arr_Country'])
from_NL_to_port3 =  collections.Counter(beyond_lasttrip[2]['ArrPort'])
from_NL_to_cntry3 =  collections.Counter(beyond_lasttrip[2]['Arr_Country'])

# Function to get top 5 + "Others"
def top5_with_others(counter_obj):
    sorted_items = sorted(counter_obj.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_items[:5]
    others_sum = sum(v for _, v in sorted_items[5:])
    top5.append(("Others", others_sum))
    return collections.Counter(dict(top5))

# Apply to all three datasets
final_result1 = top5_with_others(from_NL_to_cntry)
final_result2 = top5_with_others(from_NL_to_cntry2)
final_result3 = top5_with_others(from_NL_to_cntry3)

# --- Step 2: Combine all keys ---
all_keys = sorted(set(final_result1) | set(final_result2) | set(final_result3))

# --- Step 3: Align values ---
values1 = [final_result1.get(k, 0) for k in all_keys]
values2 = [final_result2.get(k, 0) for k in all_keys]
values3 = [final_result3.get(k, 0) for k in all_keys]

# --- Step 4: Plot ---
x = np.arange(len(all_keys))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, values1, width, label='3 ports', color='skyblue')
ax.bar(x, values2, width, label='4 ports', color='orange')
ax.bar(x + width, values3, width, label='5 ports', color='green')

# Formatting
ax.set_ylabel('Counts', fontsize=20)
ax.set_title('Top 5 returned countries after leaving NL', fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(all_keys, rotation=45, ha='right', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)

plt.tight_layout()
plt.show()

# %% Add oil import and export status

routes_w_stat_collect = []
for dfs in hotspot_toward_collect[1:]:
    routes_w_stat = []
    for df in dfs:
        df_stat = pd.merge(df, imo_w_stat, on = 'Country', how = 'inner')
        routes_w_stat.append(df_stat)
    routes_w_stat_collect.append(routes_w_stat)

count_net_importer = 0
count_net_exporter = 0
full_4 = []
empty_4 = []

for df in routes_w_stat_collect[0]:
    if df['Status'].iloc[-1] == 'Net importer':
        count_net_importer += 1
        empty_4.append(df)
    else:
        count_net_exporter += 1
        full_4.append(df)
        
# trip after NL for tankers arriving empty
oil_status = [empty_4, full_4 ]
from_NL_to = []
for lst in oil_status: 
    per_row2 = pd.DataFrame(columns = alltankers_adjusted.columns)
    for df in lst:


        last_row = df.iloc[[-1]]
        per_row2 = pd.concat([per_row2, last_row])
    per_row2 = per_row2.drop_duplicates()
    status_tanker_to = pd.DataFrame(columns = alltankers_adjusted.columns)
    for row in range(len(per_row2)):
        imo = per_row2.iloc[row]['IMO']
        imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
        look_row = per_row2.iloc[[row]]
        for nr_row in range(len(imo_poc)):
            if (look_row.iloc[0]["DepDate"] == imo_poc.iloc[nr_row]['DepDate']) & (look_row.iloc[0]["ArrDate"] == imo_poc.iloc[nr_row]['ArrDate']):
                status_tanker_to = pd.concat([status_tanker_to, imo_poc.iloc[nr_row+1:nr_row+2]])
                
    from_NL_to.append(status_tanker_to)
from_NL_to_w_status = []
for lst in from_NL_to:
    
    lst = pd.merge(lst, imo_w_stat, left_on = 'Arr_Country', right_on= 'Country', how = 'inner')
    from_NL_to_w_status.append(lst)
count_status_from_empty = collections.Counter(from_NL_to_w_status[0]['Status'])
count_status_from_full = collections.Counter(from_NL_to_w_status[1]['Status'])

# Example: counts_empty and counts_full
empty = from_NL_to_w_status[0][['Status', 'Country_y']]
full = from_NL_to_w_status[1][['Status', 'Country_y']]

counts_empty = empty.groupby(["Status", "Country_y"]).size().unstack(fill_value=0)
counts_full = full.groupby(["Status", "Country_y"]).size().unstack(fill_value=0)

# Combine columns to ensure consistent ordering
all_countries = sorted(list(set(counts_empty.columns) | set(counts_full.columns)))
counts_empty = counts_empty.reindex(columns=all_countries, fill_value=0)
counts_full = counts_full.reindex(columns=all_countries, fill_value=0)

# Ensure correct order of x-axis categories
status_order = ["Mixed", "Net exporter", "Net importer"]
counts_empty = counts_empty.reindex(status_order, fill_value=0)
counts_full = counts_full.reindex(status_order, fill_value=0)

# Set bar positions
x = np.arange(len(status_order))
width = 0.4  # width of each bar

# Assign a color to each country
colors = plt.cm.tab20.colors  # up to 20 distinct colors
country_colors = {country: colors[i % len(colors)] for i, country in enumerate(all_countries)}

fig, ax = plt.subplots(figsize=(12, 6))

# Plot stacked bars for empty
bottom_empty = np.zeros(len(status_order))
for country in all_countries:
    ax.bar(x - width/2, counts_empty[country], width, bottom=bottom_empty, label=country, color=country_colors[country])
    bottom_empty += counts_empty[country]

# Plot stacked bars for full
bottom_full = np.zeros(len(status_order))
for country in all_countries:
    ax.bar(x + width/2, counts_full[country], width, bottom=bottom_full, label=country, color=country_colors[country])
    bottom_full += counts_full[country]

# Optional: add country labels on top of each segment
for i, status in enumerate(status_order):
    bottom = 0
    for country in all_countries:
        val = counts_empty.loc[status, country]
        if val > 0:
            ax.text(x[i] - width/2, bottom + val/2, country, ha='center', va='center', fontsize=18)
            bottom += val
    bottom = 0
    for country in all_countries:
        val = counts_full.loc[status, country]
        if val > 0:
            ax.text(x[i] + width/2, bottom + val/2, country, ha='center', va='center', fontsize=18)
            bottom += val

ax.set_xticks(x)
ax.set_xticklabels(status_order, fontsize=18) 
ax.set_xticklabels(status_order, fontsize=14)  # X-axis tick labels
ax.set_xlabel("Status", fontsize=20)           # X-axis label
ax.set_ylabel("Count", fontsize=20)            # Y-axis label
ax.set_title("Returned countries from a Dutch port (empty tankers vs full tankers", fontsize=18)  # Title
# Increase tick label size for both axes
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()


count_net_importer_5 = 0
count_net_exporter_5 = 0

for df in routes_w_stat_collect[1]:
    if df['Status'].iloc[-1] == 'Net importer':
        count_net_importer_5 += 1
    else:
        count_net_exporter_5 += 1

count_net_importer_51 = 0
count_net_exporter_51 = 0

for df in routes_w_stat_collect[1]:
    if df['Status'].iloc[2] == 'Net importer' & df['Status'].iloc[3] == 'Net importer':
        count_net_importer_51 += 1
    else:
        count_net_exporter_51 += 1
        
    


count_net_importer = 0
count_net_exporter = 0
full_4 = []
empty_4 = []

for df in routes_w_stat_collect[1]:
    if df['Status'].iloc[-1] == 'Net importer':
        count_net_importer += 1
        empty_4.append(df)
    else:
        count_net_exporter += 1
        full_4.append(df)
        
# trip after NL for tankers arriving empty
oil_status = [empty_4, full_4 ]
from_NL_to = []
for lst in oil_status: 
    per_row2 = pd.DataFrame(columns = alltankers_adjusted.columns)
    for df in lst:


        last_row = df.iloc[[-1]]
        per_row2 = pd.concat([per_row2, last_row])
    per_row2 = per_row2.drop_duplicates()
    status_tanker_to = pd.DataFrame(columns = alltankers_adjusted.columns)
    for row in range(len(per_row2)):
        imo = per_row2.iloc[row]['IMO']
        imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
        look_row = per_row2.iloc[[row]]
        for nr_row in range(len(imo_poc)):
            if (look_row.iloc[0]["DepDate"] == imo_poc.iloc[nr_row]['DepDate']) & (look_row.iloc[0]["ArrDate"] == imo_poc.iloc[nr_row]['ArrDate']):
                status_tanker_to = pd.concat([status_tanker_to, imo_poc.iloc[nr_row+1:nr_row+2]])
                
    from_NL_to.append(status_tanker_to)
from_NL_to_w_status = []
for lst in from_NL_to:
    
    lst = pd.merge(lst, imo_w_stat, left_on = 'Arr_Country', right_on= 'Country', how = 'inner')
    from_NL_to_w_status.append(lst)
count_status_from_empty = collections.Counter(from_NL_to_w_status[0]['Status'])
count_status_from_full = collections.Counter(from_NL_to_w_status[1]['Status'])

# Example: counts_empty and counts_full
empty = from_NL_to_w_status[0][['Status', 'Country_y']]
full = from_NL_to_w_status[1][['Status', 'Country_y']]

counts_empty = empty.groupby(["Status", "Country_y"]).size().unstack(fill_value=0)
counts_full = full.groupby(["Status", "Country_y"]).size().unstack(fill_value=0)

# Combine columns to ensure consistent ordering
all_countries = sorted(list(set(counts_empty.columns) | set(counts_full.columns)))
counts_empty = counts_empty.reindex(columns=all_countries, fill_value=0)
counts_full = counts_full.reindex(columns=all_countries, fill_value=0)

# Ensure correct order of x-axis categories
status_order = ["Mixed", "Net exporter", "Net importer"]
counts_empty = counts_empty.reindex(status_order, fill_value=0)
counts_full = counts_full.reindex(status_order, fill_value=0)

# Set bar positions
x = np.arange(len(status_order))
width = 0.4  # width of each bar

# Assign a color to each country
colors = plt.cm.tab20.colors  # up to 20 distinct colors
country_colors = {country: colors[i % len(colors)] for i, country in enumerate(all_countries)}

fig, ax = plt.subplots(figsize=(12, 6))

# Plot stacked bars for empty
bottom_empty = np.zeros(len(status_order))
for country in all_countries:
    ax.bar(x - width/2, counts_empty[country], width, bottom=bottom_empty, label=country, color=country_colors[country])
    bottom_empty += counts_empty[country]

# Plot stacked bars for full
bottom_full = np.zeros(len(status_order))
for country in all_countries:
    ax.bar(x + width/2, counts_full[country], width, bottom=bottom_full, label=country, color=country_colors[country])
    bottom_full += counts_full[country]

# Optional: add country labels on top of each segment
for i, status in enumerate(status_order):
    bottom = 0
    for country in all_countries:
        val = counts_empty.loc[status, country]
        if val > 0:
            ax.text(x[i] - width/2, bottom + val/2, country, ha='center', va='center', fontsize=18)
            bottom += val
    bottom = 0
    for country in all_countries:
        val = counts_full.loc[status, country]
        if val > 0:
            ax.text(x[i] + width/2, bottom + val/2, country, ha='center', va='center', fontsize=18)
            bottom += val

ax.set_xticks(x)
ax.set_xticklabels(status_order, fontsize=18) 
ax.set_xticklabels(status_order, fontsize=14)  # X-axis tick labels
ax.set_xlabel("Status", fontsize=20)           # X-axis label
ax.set_ylabel("Count", fontsize=20)            # Y-axis label
ax.set_title("Returned countries from a Dutch port (empty tankers vs full tankers", fontsize=18)  # Title
# Increase tick label size for both axes
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.show()
# 3 ports
# extract and analyze the first IMO from RU per trip


# %% OLD CODE
# sorted_items = sorted(counter_ctry_arr3.items(), key=lambda x: x[1], reverse=True)

# # Step 3: Take top 5
# top5 = sorted_items[:5]

# # Step 4: Sum the rest as "Others"
# others_sum = sum(v for _, v in sorted_items[5:])
# top5.append(("Others", others_sum))

# # Step 5: Convert back to Counter or keep as list
# final_result = collections.Counter(dict(top5))

# # Get all unique keys from all counters
# keys = sorted(set(final_result))
# values = [final_result.get(k, 0) for k in keys]

# # Bar width and positions
# bar_width = 0.25
# x = range(len(keys))
# plt.bar(x, values, width=bar_width, label='Counter 2')

# # Labels and title
# plt.xticks(x, keys, rotation=90)  # Rotate labels vertically
# plt.ylabel('Count')
# plt.title('Comparison of Counters')
# plt.legend()

# plt.show()

# max_size = max(len(inner) for inner in route5)


# hotspot_toward_collect = []

# for n in range(4, max_size+1):


#             route_RU_int_NL_matched_imoNr2,trip_freq_dict2, country_seq2, port_sequence2 = pr.route_seq_matched_nrimo_par(
#                 route5, alltankers_adjusted, n, 3,  False, oiltype = 'all', loop_type = 'country')
#             route_dir2 = []
#             # check how many extracted route matched with the reported suspecious routes
#             for df in route_RU_int_NL_matched_imoNr2:
#                 if df.iloc[0, df.columns.get_loc('Arr_Country')] in (country_of_interest):
#                            route_dir2.append(df)
#             hotspot_toward = []
            

#             for df in route_dir2:
#                 if df['IMO'].iloc[0] != df['IMO'].iloc[1]:
#                     hotspot_toward.append(df)
#             hotspot_toward_collect.append(hotspot_toward)

    
# # TOTAL 5 PORTs

# route5 = joblib.load('./processing/pr_inter_output/potential_routes_loop_nrRU_1_time2w_nrtotport5.joblib')
# route_RU_int_NL_matched_imoNr5,trip_freq_dict5, country_seq5, port_sequence5 = pr.route_seq_matched_nrimo_par(
#     route5, alltankers_adjusted, 5, 3,  False, oiltype = 'all', loop_type = 'country')


# route_dir2 = []
# # check how many extracted route matched with the reported suspecious routes
# for df in route_RU_int_NL_matched_imoNr5:
#     if df.iloc[0, df.columns.get_loc('Arr_Country')] in (country_of_interest):
#                route_dir2.append(df)
# perc_route_matched_reported2 = (len(route_dir2)/len(route_RU_int_NL_matched_imoNr5))*100    

# # extract and analyze the IMO per row or per trip
# per_row2 = pd.DataFrame(columns = alltankers_adjusted.columns)
# for df in route_dir2:
#     first_row = df[0:1]
#     per_row2 = pd.concat([per_row2, first_row])
    
# # drop dubplicated
# next_trip2 = pd.DataFrame(columns = alltankers_adjusted.columns)
# per_row2 = per_row2.drop_duplicates()
# for row in range(len(per_row2)):
#     imo = per_row2.iloc[row]['IMO']
#     imo_poc = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
#     row_index = per_row2.iloc[row:row+1].index[0] 
#     for nr_row in range(len(imo_poc)):
#         if row_index == imo_poc.iloc[nr_row:nr_row+1].index[0]:
#             next_trip2 = pd.concat([next_trip2, imo_poc.iloc[nr_row+1:nr_row+2]])
#             continue
# # # check if there is any same set of imo occurs

# country_pair2 = tuple(zip(next_trip2['Country'], next_trip2['Arr_Country']))
# counter_cntr_pair2 = collections.Counter(country_pair2)
# counter_ctry_arr2 = collections.Counter(next_trip2['Arr_Country'])

# # select trips have two different consecutive IMO at the first share ports of the routes
# hotspot_toward = []
# for df in route_dir2:
#     if df['IMO'].iloc[0] != df['IMO'].iloc[1]:
#         hotspot_toward.append(df)
# snd_trip = pd.DataFrame(columns = alltankers_adjusted.columns)

# for df in hotspot_toward:
#     second_trip = df.iloc[1:2]
#     snd_trip = pd.concat([snd_trip, second_trip])
    
# snd_trip = snd_trip.drop_duplicates()
# country_pair3 = tuple(zip(snd_trip['Country'], snd_trip['Arr_Country']))
# counter_cntr_pair3 = collections.Counter(country_pair3)
# counter_ctry_arr3 = collections.Counter(snd_trip['Arr_Country'])

# #Before NL
# thrd_trip = pd.DataFrame(columns = alltankers_adjusted.columns)
# for df in hotspot_toward:
#     third_trip = df.iloc[2:3]
#     thrd_trip = pd.concat([thrd_trip, third_trip])
    
# thrd_trip = thrd_trip.drop_duplicates()
# country_pair4 = tuple(zip(thrd_trip['Country'], thrd_trip['Arr_Country']))
# counter_cntr_pair4 = collections.Counter(country_pair4)
# counter_ctry_arr4 = collections.Counter(thrd_trip['Arr_Country'])

# sum_two =  counter_ctry_arr4 +   counter_ctry_arr3 
