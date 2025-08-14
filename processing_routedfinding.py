# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:54:41 2025

@author: Duyen
"""
import os
cwd = os.getcwd()
os.chdir(cwd)

from itertools import islice
import sys
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import itertools
from datetime import datetime, timedelta
import networkx as nx
from Code import data_processing as pr
from Code import data_preprocessing as pp
import numpy as np
import pandas as pd
import psutil
import joblib

import multiprocessing

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# Create directory to story data
pr_input_path = './processing/pr_inter_input'
pr_output_path = './processing/pr_inter_output'
try:
    os.makedirs(pr_output_path)
    os.makedirs(pr_input_path)
    print(f" path '{pr_input_path}' created successfully")
    print(f" path '{pr_output_path}' created successfully")
except FileExistsError:
    print(
        f"One or more direcotries in '{pr_input_path}' and '{pr_output_path}' aldready exist")

except PermissionError():
    print(
        f"Permission denied: Unable to create '{pr_input_path}' and '{pr_output_path}'")
except Exception as e:
    print(f"An error occured: {e}")
# %% import data
alltankers = pd.read_csv(
    './preprocessing/inter_input/All Port calls - NL & RU.csv')

# select data from NL and RU ww only
alltankers = alltankers[alltankers['PATH'].isin(
    ['Tankers were to NL (worldwide)', 'Tankers were to RU (worldwide)'])]
# %% preprocessing data
alltankers = alltankers[['IMO', 'SHIPTYPE',
                         'COUNTRY ', 'PORTNAME', 'ARRIVALDATE', 'SAILDATE']]
alltankers = alltankers.rename(columns={'SAILDATE': 'DEPDATE'})
portname = alltankers['PORTNAME'].drop_duplicates()
# remove dublicate
alltankers = alltankers.drop_duplicates()
# standadize ship name
alltankers['SHIPTYPE'] = alltankers['SHIPTYPE'].map(
    lambda x: pp.standadize_ship_type(x))
# convert columns to the right format
alltankers['ARRIVALDATE'] = alltankers['ARRIVALDATE'].astype('datetime64[ns]')
alltankers['DEPDATE'] = alltankers['DEPDATE'].astype('datetime64[ns]')
# calculate time a vessel spent in a port for each POC
seconds_in_day = 24*60*60
alltankers['TIMEINPORT'] = alltankers['DEPDATE'] - alltankers['ARRIVALDATE']
# calculate time a vessel travel from one port to another port

alltankers_adjusted = pd.DataFrame()
for imo in alltankers['IMO'].unique():
    a_imo = alltankers[alltankers['IMO'] == imo]
    Port2Port = pd.DataFrame()
    Port2Port['IMO'] = a_imo['IMO']
    Port2Port['DepPort'] = a_imo['PORTNAME']
    Port2Port['ArrPort'] = a_imo['PORTNAME'].shift(-1)
    Port2Port['DepDate'] = a_imo['DEPDATE']
    Port2Port['ArrDate'] = a_imo['ARRIVALDATE'].shift(-1)
    Port2Port['ShipType'] = a_imo['SHIPTYPE']
    Port2Port['Country'] = a_imo['COUNTRY ']
    Port2Port['TravelTime'] = abs(Port2Port['DepDate'] - Port2Port['ArrDate'])
    Port2Port['BerthTime'] = Port2Port['DepDate'].shift(
        -1) - Port2Port['ArrDate']
    alltankers_adjusted = pd.concat([alltankers_adjusted, Port2Port])

# remove row that contain Nan
alltankers_adjusted = alltankers_adjusted.dropna(subset=['DepPort', 'ArrPort'])

# # standadize port name
alltankers_adjusted['DepPort'] = alltankers_adjusted['DepPort'].map(lambda x:
                                                                    pp.standardize_port_name(x))
alltankers_adjusted['ArrPort'] = alltankers_adjusted['ArrPort'].map(lambda x:
                                                                    pp.standardize_port_name(x))
# adding arrive country column
port_itscountry = alltankers_adjusted[['DepPort', 'Country']]
port_itscountry = port_itscountry.drop_duplicates()
alltankers_adjusted = pd.merge(
    alltankers_adjusted, port_itscountry, left_on='ArrPort', right_on='DepPort')
alltankers_adjusted = alltankers_adjusted.rename(columns={
    'Country_x': 'Country', 'Country_y': 'Arr_Country', 'DepPort_x': 'DepPort'})
alltankers_adjusted = alltankers_adjusted.drop('DepPort_y', axis=1)
# Remove all domestic tankers and tanker with non crude or refined oil
remove_shiptype = ['Asphalt/Bitumen Tanker', 'Oil Bunkering Tanker', 'Shuttle Tanker', 
                   'Oil Bunkering Tanker (Inland)']
alltankers_adjusted = alltankers_adjusted[
    ~alltankers_adjusted['ShipType'].isin(remove_shiptype)]
interl_imo = []
for imo in alltankers_adjusted['IMO'].unique():
    df =  alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    if len(df['Country'].unique()) != 1:
        interl_imo.append(imo)
    
alltankers_adjusted = alltankers_adjusted[
    alltankers_adjusted['IMO'].isin(interl_imo)]    

# combine time for the same ports

mask_dub = alltankers_adjusted['DepPort'] != alltankers_adjusted['ArrPort']
alltankers_adjusted =   alltankers_adjusted[mask_dub]  
# write  csv
alltankers_adjusted.to_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv', index = True)  

# %% Port selections for different regions

country_of_interest = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Kazakhstan', 'Russia', 'Netherlands']
ports_by_country = {}
for ctry in country_of_interest:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    ports_by_country[ctry] = list(ctry_ports)

# Select RU port

ru_country = ['Russia']
port_of_russia = []
for nr in range(len(alltankers_adjusted)):
    if alltankers_adjusted.iloc[nr, 6] in ru_country:
        ruport = alltankers_adjusted.iloc[nr, 1]
        port_of_russia.append(ruport)
    else:
        next
port_of_russia = list(set(port_of_russia))
len(port_of_russia)
# Select EU countries
eu_countries = [
    "Netherlands","Sweden",  "Belgium", "Greece", "Italy", "France", "Spain",
    "Germany", "Finland", "Poland", "Denmark", "Portugal", "Romania", "Lithuania",
    "Ireland", "Malta", "Cyprus", "Bulgaria", "Croatia", "Slovenia", "Estonia",
    "Latvia"]
# Select NL country
NL = ['Netherlands']
# Select EU ports within EU countries

eu_ports = []
for nr in range(len(alltankers_adjusted)):
    if alltankers_adjusted.iloc[nr, 6] in eu_countries:
        euport = alltankers_adjusted.iloc[nr, 1]
        eu_ports.append(euport)
    else:
        next
eu_ports = list(set(eu_ports))
# Select the Dutch ports

NL_ports = []
for nr in range(len(alltankers_adjusted)):
    if alltankers_adjusted.iloc[nr, 6] in NL:
        NLport = alltankers_adjusted.iloc[nr, 1]
        NL_ports.append(NLport)
    else:
        next
        
NL_ports = list(set(NL_ports))
# %% Creating network and find neighbours and connected IMO
# create network
start_time = time.time()

network_edges = []
for n in range(len(alltankers_adjusted)):
    info = tuple([alltankers_adjusted['DepPort'].iloc[n], 
                  alltankers_adjusted['ArrPort'].iloc[n],
                  {'DepDate' : str(alltankers_adjusted['DepDate'].iloc[n]),
                   'ArrDate' : str(alltankers_adjusted['ArrDate'].iloc[n]),
                   'TravelTime' : str(alltankers_adjusted['TravelTime'].iloc[n]),
                   'IMO': alltankers_adjusted['IMO'].iloc[n]}])
    network_edges.append(info)
# create graph
## multi-direct-graph
Graph_whole_dataset = nx.MultiDiGraph()
Graph_whole_dataset.add_edges_from(network_edges)
# betweeness centrality
btwcentr = nx.betweenness_centrality(Graph_whole_dataset)
## direct-graph
direct_graph = nx.DiGraph()
direct_graph.add_edges_from(network_edges)
# Create all combination of RU and NL ports
comb_ru_nl_ports = list(itertools.product(port_of_russia, NL_ports))
# extract betweeness centrality of ports for each country
bwtcentr_ports = []
for ctr_of_int in country_of_interest:
    filter_value = {port: btwcentr[port] for port in ports_by_country[ctr_of_int]}
    # Get top 2 keys with highest values
    top_2_keys = sorted(filter_value, key=filter_value.get, reverse=True)[:2]
    bwtcentr_ports.append(top_2_keys)
bwtcentr_ports = [port for sublist in bwtcentr_ports for port in sublist]

bwtcentr_w_RUport = ['Novorossiysk','Ust Luga']
bwtcentr_w_NLport = ['Amsterdam', 'Rotterdam']
bwtcentr_ports.remove('Novorossiysk')
bwtcentr_ports.remove('Ust Luga')
bwtcentr_ports.remove('Amsterdam')
bwtcentr_ports.remove('Rotterdam')
# %% start extracting route
# extract route for one stop
# start port in RU
# threshold time gap in hours
start_time = time.time()
up_t_time = 24*7#float('inf')
low_t_time = 0
scnd_in_day = 1*24*60*60
# Round 1
# iteration value
n = 0
m = 8
# start iterating
start_RU_port = ['Novorossiysk']
nbs_edges_RU_cmb = []
nbs_edges_RU = []
filtered_final_route_RU_to_NL = []
if len(start_RU_port) == 1:
    nbs_edges_RU = list(set(list(Graph_whole_dataset.out_edges(start_RU_port,))))
    
else:
    for ruport in start_RU_port:
        nbs_edges = list(set(list(Graph_whole_dataset.out_edges(ruport,))))
        nbs_edges_RU_cmb.append(nbs_edges)

# unlist nbs edges
if len(nbs_edges_RU_cmb) > 1:
    for lst in nbs_edges_RU_cmb:
        for sublst in lst:
            nbs_edges_RU.append(sublst)

# loop through all neighbours of Novorossiysk


track_route_fr_RU_to_2ndPort_and_connected_IMO = pr.find_matched_imo_at_1stshared_port(
    nbs_edges_RU, port_of_russia,
    eu_ports, alltankers_adjusted, 
    scnd_in_day, low_t_time, up_t_time)
# extract route from RU-hotpot-NL and number of IMO difference            
route_RU_1int_NL = []
route_RU_1int_other = []
# extract route from RU-aport-NL
for df in track_route_fr_RU_to_2ndPort_and_connected_IMO:
    info_shared_port = df
    if (info_shared_port['ArrPort'].isin(bwtcentr_w_NLport)).any():
        if (info_shared_port['DepPort'].isin(bwtcentr_ports)).any():
            
            route_RU_1int_NL.append(info_shared_port)
            
    else:
        route_RU_1int_other.append(info_shared_port)
#* save final route from RU to NL
filtered_final_route_RU_to_NL.append(route_RU_1int_NL)      
# select for the total number of IMO
route_RU_1int_NL_matched_imoNr = []
for df in route_RU_1int_NL:
    info_shared_port = df
    if len(info_shared_port['IMO'].unique()) == 2: # fill in TOTAL NR. OF PORT ALLOWED
        route_RU_1int_NL_matched_imoNr.append(info_shared_port)
        
# calculate unique route frequency
port_sequence = {}
for df in route_RU_1int_NL_matched_imoNr:

    port_list = df['DepPort'].tolist()
    port_list.append(df.iloc[-1]['ArrPort'])
    port_list = tuple(port_list)
    if port_list not in list(port_sequence.keys()):
        port_sequence[port_list] =  1
    else:
        port_sequence[port_list] = port_sequence.get(port_list) + 1

# iteration calculation
n = n+1
# Round 2:
# get nbs of the next port
route_RU_to_NL = route_RU_1int_other
while n < m:

    
    track_route_fr_RU_to_NL = pr.find_matched_imo_at_shared_port(route_RU_to_NL,alltankers_adjusted, 
                                       df,
                                      scnd_in_day,
                                      low_t_time,
                                      up_t_time)
        
        
    route_RU_int_NL, route_RU_int_other = pr.extract_route_RU_to_NL_and_others(
        track_route_fr_RU_to_NL,
        bwtcentr_w_NLport,
        bwtcentr_ports)
    route_RU_int_NL_filtered_v1 = []
    for df in route_RU_int_NL:
        lst_locations = df.iloc[-1]
        if lst_locations['DepPort'] not in port_of_russia:
            route_RU_int_NL_filtered_v1.append(df)
    
    
    # next port(eu) to NL
    route_RU_int_NL_filtered_v2 = []
    
    for df in route_RU_int_NL_filtered_v1:
    
        if df['DepPort'].isin(eu_ports).any():
            depPort = np.array(df['DepPort'])
            mask_euport = np.isin(depPort, eu_ports)
            ind = np.where(mask_euport)[0].tolist()
            if len(ind) == 1:
                if df['IMO'].iloc[ind[0]] == df['IMO'].iloc[(ind[0]-1)]:
                    route_RU_int_NL_filtered_v2.append(df)
                else:
                    next
            else:
                boo_check = []
                for indx in ind:
                    if df['IMO'].iloc[indx] == df['IMO'].iloc[(indx -1)]:
                        if indx < len(df)-1:
                            if df['IMO'].iloc[indx] == df['IMO'].iloc[(indx + 1)]:
                                boo_check.append(True)
                            else:
                                boo_check.append(False)
                                
                        else:
                            boo_check.append(True)
                    else:
                        boo_check.append(False)
                if len(boo_check) > 0 and all(boo_check):
                    route_RU_int_NL_filtered_v2.append(df)
                        
        else: 
            route_RU_int_NL_filtered_v2.append(df)
     #* save final routes from RU to NL
    filtered_final_route_RU_to_NL.append(route_RU_int_NL_filtered_v2)    

         
    # update list of routes for the next round iteration
    route_RU_to_NL = route_RU_int_other
    print('iter:', n)
    print('used mem:', psutil.virtual_memory().percent)
    run_time = (time.time() - start_time)
    print('run time:', run_time)
    n = n+1
    


# save result
# save list of dataframe
with open('./processing/pr_inter_output/route_w_1w_7m.pkl', 'wb') as outp:
    pickle.dump(filtered_final_route_RU_to_NL, outp, pickle.HIGHEST_PROTOCOL)   


# Save
joblib.dump(filtered_final_route_RU_to_NL, './processing/pr_inter_output/route_w_1w_7m.joblib')
# you can have the percentage of used RAM
used_mem = psutil.virtual_memory().percent

# you can calculate percentage of available memory
avai_mem = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
save_text = 'run_time:' + str(run_time) + 'used_mem:' + str(used_mem) + 'avail_mem:' + str(avai_mem)
with open("./processing/pr_inter_output/run_time_1w8m.txt", "w") as f:
    f.write(save_text)
    
    
#This will store a list of all the variables in the program
d = dir()

#You'll need to check for user-defined variables in the directory
for obj in d:
    #checking for built-in variables/functions
    if not obj.startswith('__'):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
#%% extract routes further based on requirement dsuch as nr of ports and imo
# final_route_RU_to_NL = []
# for lst in filtered_final_route_RU_to_NL:
#     for sublst in lst:

#             final_route_RU_to_NL.append(sublst)
# print("--- %s seconds ---" % (time.time() - start_time))
# # select for the total number of IMO
# route_RU_int_NL_matched_imoNr = pr.route_seq_matched_nrimo(
#     final_route_RU_to_NL, 4, 3)

# # calculate unique route frequency
# port_sequence = pr.freq_of_port_seq(route_RU_int_NL_matched_imoNr)
#%% Analysis
# calculate the total time for a complete route from RU to NL
# time_trvl_in_mth = []
# for df in route_RU_int_NL_matched_imoNr:

#     travel_dur = abs(df['DepDate'].iloc[0] - df['ArrDate'].iloc[-1])
#     travel_dur = round((travel_dur.days * seconds_in_day 
#                         + travel_dur.seconds)/(24*3600*30.47), 0)
#     time_trvl_in_mth.append(travel_dur)

# time_trvl_uniq = list(set(time_trvl_in_mth))
# time_trvl_dict = {item: time_trvl_in_mth.count(item) for item in time_trvl_uniq}

# # plot
# plt.bar(range(len(time_trvl_dict)), list(time_trvl_dict.values()), align = 'center')
# plt.xticks(range(len(time_trvl_dict)), list(time_trvl_dict.keys()))
# plt.ylabel('Nr. of routes')
# plt.xlabel('Travel time (in month)')
# plt.title('Travel duration starting from a RU port to a NL port (total port: ? and total imo:?)')
# plt.show() # ADD total port and imo
# # check the routes that have travel duration time smaller than a certain number
# route_w_certain_trvl_dur = []
# for df in route_RU_int_NL_matched_imoNr:

#     travel_dur = abs(df['DepDate'].iloc[0] - df['ArrDate'].iloc[-1])
#     travel_dur = round((travel_dur.days * seconds_in_day 
#                         + travel_dur.seconds)/(24*3600*30.47), 0)
#     if travel_dur <= 7 and travel_dur >= 6: # ADD certain month for checking
#         route_w_certain_trvl_dur.append(df)

# #  calculate the frequency of imo involved in the potential RU oil transportation
# # in a given nr of ports and nr of imo
# route_RU_int_NL_matched_imoNr_merge = pd.concat(route_RU_int_NL_matched_imoNr, 
#                                                 ignore_index= True)
  
# imo_array = np.concatenate([df['IMO'].unique() for df in route_RU_int_NL_matched_imoNr]).tolist()
# imo_uniq = list(set(imo_array))
# imo_dict = {item: imo_array.count(item) for item in imo_uniq}
# check_route_w_pp_imo = [9319686]
# route_w_pp_imo = [df for df in route_RU_int_NL_matched_imoNr if df['IMO'].isin(check_route_w_pp_imo).any()]
# # check if there is any same set of imo occurs
# set_imo = [tuple(df['IMO'].unique()) for df in route_RU_int_NL_matched_imoNr]
# set_imo_uniq = np.unique(set_imo)
# set_imo_dict = {item: set_imo.count(item) for item in imo_uniq}
# counter = collections.Counter(set_imo)

# check_route_w_pp_imo = [9319686, 9346720, 9767340 ]
# route_w_pp_imo_seq = [df for df in route_RU_int_NL_matched_imoNr if 
#                   np.all(np.isin(df['IMO'].unique(),check_route_w_pp_imo))]
a = alltankers_adjusted[alltankers_adjusted['IMO']==9832547]
