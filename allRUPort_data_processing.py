# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:22:15 2025

@author: Duyen
"""
import pandas as pd
import os
os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
import numpy as np
from Code import data_preprocessing as pp
from Code import data_processing as pr
import re
import csv
import networkx as nx
from datetime import datetime, timedelta
from itertools import islice  
import collections
import itertools
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import sys
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
    print(f"One or more direcotries in '{pr_input_path}' and '{pr_output_path}' aldready exist")
    
except PermissionError():
    print(f"Permission denied: Unable to create '{pr_input_path}' and '{pr_output_path}'")
except Exception as e:
    print(f"An error occured: {e}")
# %% import data
alltankers = pd.read_csv('./preprocessing/inter_input/All Port calls - NL & RU.csv')

# select data from NL and RU ww only
alltankers = alltankers[alltankers['PATH'].isin(['Tankers were to NL (worldwide)', 'Tankers were to RU (worldwide)'])]
alltankers = alltankers[['IMO', 'SHIPTYPE', 'COUNTRY ', 'PORTNAME', 'ARRIVALDATE', 'SAILDATE']]
alltankers = alltankers.rename(columns = {'SAILDATE' : 'DEPDATE'})
portname = alltankers['PORTNAME'].drop_duplicates()
# remove dublicate
alltankers = alltankers.drop_duplicates()
# standadize ship name
alltankers['SHIPTYPE'] = alltankers['SHIPTYPE'].map(lambda x: pp.standadize_ship_type(x))
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
    Port2Port['BerthTime'] = Port2Port['DepDate'].shift(-1) - Port2Port['ArrDate']
    alltankers_adjusted = pd.concat([alltankers_adjusted,Port2Port])
    
# remove row that contain Nan
alltankers_adjusted = alltankers_adjusted.dropna(subset = ['DepPort', 'ArrPort'])
# # sort values to depature date
# alltankers_adjusted = alltankers_adjusted.sort_values(by = 'DepDate')
# # Save to CSV
# alltankers_adjusted_ch = alltankers_adjusted
# alltankers_adjusted_ch = alltankers_adjusted_ch.rename(columns={'DepPort': 'Source', 'ArrPort':'Target'})
# alltankers_adjusted_ch.to_csv('./preprocessing/pp_inter_ouput/alltanker.csv', index=False)
# # Save to CSV
# alltankers_adjusted_selec = alltankers_adjusted[['IMO', 'DepPort']]

# alltankers_adjusted_selec.columns = ['IMO', 'ID']
# alltankers_adjusted_selec['Name'] = alltankers_adjusted_selec['ID']
# alltankers_adjusted_selec = alltankers_adjusted_selec.drop_duplicates(subset='ID')
# alltankers_adjusted_selec.to_csv('./preprocessing/pp_inter_ouput/node.csv', index=False)

# # standadize port name
alltankers_adjusted['DepPort'] = alltankers_adjusted['DepPort'].map(lambda x: 
                                                                    pp.standardize_port_name(x))
alltankers_adjusted['ArrPort'] = alltankers_adjusted['ArrPort'].map(lambda x: 
                                                                pp.standardize_port_name(x))

port_itscountry = alltankers_adjusted[['DepPort', 'Country']]
port_itscountry = port_itscountry.drop_duplicates()
# %% Port selections for different regions
# select refinery hubs    
ports_of_interest = [
    # India
    "Sikka", "Mumbai", "Paradip", "Deendayal", "Chennai", "Mundra", "Haldia", "Kattupalli", "Jawaharlal Nehru Port",

    # China
    "Qinzhou", "Zhoushan", "Zhanjiang", "Dongshan", "Tianjin", "Qingdao", "Ningbo", "Dongjiakou", "Caofeidian",
    "Lianyungang", "Shidao", "Dalian", "Yangpu", "Yantai", "Huizhou", "Shekou",

    # Türkiye (Turkey)
    "Marmara Ereglisi Terminals", "Ceyhan", "Mersin", "Aliaga", "Dortyol", "Diliskelesi", "Yarimca", "Istanbul", "Yalova",

    # United Arab Emirates (UAE)
    "Jebel Ali", "Fujairah", "Ruwais", "Sharjah", "Zirku Island", "Das Island", "Port Rashid",

    # Singapore
    "Singapore",

    # Malaysia
    "Pengerang Terminal", "Tanjung Pelepas", "Port Dickson", "Johor", "Sungai Udang", "Port Klang", "Tanjung Uban", "Malacca",

    # Indonesia
    "Balongan", "Dumai", "Tanjung Intan", "Tanjung Balai Karimun",

    # Vietnam
    "Dung Quat", "Van Phong Bay", "FPSO 'Thai Binh-VN'",

    # Saudi Arabia
    "Rabigh", "Jeddah", "King Fahd Industrial Port (Yanbu)", "Ras Tanura", "Jubail", "Jizan", "Ras Al Khafji",

    # South Africa
    "Richards Bay", "Cape Town", "Durban", "Saldanha Bay"
]
country_of_interest = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Kazakhstan', 'Russia']
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
filter_value = {port: btwcentr[port] for port in ports_by_country['Korea (South)']}

# define selfloop edge and remove it
 # define and remove nodes with only selfloop
Graph_whole_dataset.remove_edges_from(list(nx.selfloop_edges(Graph_whole_dataset)))
self_loops = list(nx.selfloop_edges(Graph_whole_dataset, keys=True))

# extract route from RU for certain IMO
# create a lookup table for ships that in Eu ports
IMO_in_EU = alltankers_adjusted[alltankers_adjusted['Country'].isin(eu_countries)]
# create a lookup table for ships that were to NL
IMO_in_NL = alltankers_adjusted[alltankers_adjusted['Country'].isin(NL)]

# threshold time gap in hours
up_t_time = float('inf')
low_t_time = 0
scnd_in_day = 1*24*60*60
# extract 1 hop
# aim of this task is to find a connection trip at the nb port of the first trip
# extract edges from st.Peterburg and Novorossiysk
# neighbors_Nov = list(Graph_whole_dataset.neighbors('Novorossiysk'))
# filter_neighbors_Nov = [n for n in neighbors_Nov if n not in eu_ports]
# out_edge_node_neighbour = []
# extract only neighbours that belongs to the routes going from Novorossiysk 
# set iteration time
m = 1
n = 0

# extract route start from all RU ports
path_frRUPort_to_nbs = alltankers_adjusted[alltankers_adjusted['Country'].isin(['Russia'])]
mask_notinEUandRUPorts = (
    (~path_frRUPort_to_nbs['Arr_Country'].isin(eu_countries)) &
    (~path_frRUPort_to_nbs['Arr_Country'].isin(['Russia']))
)
# remove destination in EU and RU
path_frRUPort_to_nonEUenRUnbs = path_frRUPort_to_nbs[mask_notinEUandRUPorts]
# extract all IMO available at the first destination
nbs = path_frRUPort_to_nonEUenRUnbs['ArrPort'].unique()
mask_nbs = alltankers_adjusted['DepPort'].isin(nbs)
#* all IMO available at the second port
paths_fr2ndPort_to_nextPort = alltankers_adjusted[mask_nbs]
# track the next neigbour of the 2nd port (hotspot)
tophotspot = ['India', 'Turkey', 'Kazakhstan', 'Brazil', 'Egypt', 'China',
              'Singapore', 'United Arab Emirates', 'Saudi Arabia', 'Korea (South)',
              'Malaysia', 'Azerbaijan']
#* select only arrival ports from RU that in the hotspot list
mask_tophotspots = path_frRUPort_to_nonEUenRUnbs['Arr_Country'].isin(tophotspot)
path_frRUPort_to_tophotspot = path_frRUPort_to_nonEUenRUnbs[mask_tophotspots]

# count and plots countries and ports exporting to the Netherlands
count_ctry_path_frRUPort_to_hotspotport = collections.Counter(path_frRUPort_to_tophotspot['Arr_Country'])
count_RUport = collections.Counter(path_frRUPort_to_tophotspot['DepPort'])
count_port_path_frRUPort_to_hotspotport = collections.Counter(path_frRUPort_to_tophotspot['ArrPort'])
#* extract route from 2nd port direct to NL
mask_NLport= paths_fr2ndPort_to_nextPort['ArrPort'].isin(NL_ports)
paths_fr2ndPort_to_NL = paths_fr2ndPort_to_nextPort[mask_NLport]
#* extract route from 2nd hotspot direct to NL
mask_hotspot_to_NL = paths_fr2ndPort_to_NL['Country'].isin(tophotspot)
paths_fr2ndhotspotPort_to_NL = paths_fr2ndPort_to_NL[mask_hotspot_to_NL]
# count and plots countries and ports exporting to the Netherlands
count_ctry_path_2ndPort_t0_NL = collections.Counter(paths_fr2ndhotspotPort_to_NL['Country'])
count_port_path_2ndPort_t0_NL = collections.Counter(paths_fr2ndhotspotPort_to_NL['DepPort'])
# count frequency the same IMO occur (direct from 2ndport to NL)
count_IMO_frRUPort_to_nonEUenRUnbs =  path_frRUPort_to_nonEUenRUnbs['IMO'].value_counts().reset_index()
count_IMO_fr2ndPort_to_NL = paths_fr2ndhotspotPort_to_NL['IMO'].value_counts().reset_index()
count_ctry_path_2ndPort_t0_NL = collections.Counter(paths_fr2ndPort_to_NL['Country'])
#* retreat path from RU-hotspot-NL
# select the most popular paths
depPort_at_NL = set(paths_fr2ndhotspotPort_to_NL['Country'])
full_routes_fr_RU_to_NL = []


   
paths_frB_toC = paths_fr2ndhotspotPort_to_NL
port_at_B = set(paths_frB_toC['DepPort'])

# Convert to DataFrame for better viewing or plotting
NL_path_likelyhood = pr.path_likelyhood(paths_frB_toC)

paths_frA_toB = path_frRUPort_to_tophotspot
RU_hotpotpath_likelyhood = pr.path_likelyhood(paths_frA_toB)


frB_toC_portpairs = zip(paths_frB_toC['DepPort'],
                        paths_frB_toC['ArrPort'])
frB_toC_portpairs_count = collections.Counter(list(frB_toC_portpairs))

    # Total number of such pairs
total_pairs = sum(frB_toC_portpairs_count.values())

# Calculate percentage for each pair
percentage_dict = {
    pair: (count / total_pairs)
    for pair, count in frB_toC_portpairs_count.items()
}
sum(percentage_dict.values())
# Convert to DataFrame for better viewing or plotting
a_path_likelyhood = pd.DataFrame(
    [(dep, arr, pct) for (dep, arr), pct in percentage_dict.items()],
    columns=["DepPort", "ArrPort", "Percentage"]
).sort_values(by="Percentage", ascending=False)

    # Normalize percentages to probabilities
RU_hotpotpath_likelyhood["Probability"] = RU_hotpotpath_likelyhood["Percentage"]
NL_path_likelyhood["Probability"] = NL_path_likelyhood["Percentage"]

# Combine segments where seg1.ArrPort == seg2.DepPort
combined_routes = []

for i, row1 in RU_hotpotpath_likelyhood.iterrows():
    for j, row2 in NL_path_likelyhood.iterrows():
        if row1["ArrPort"] == row2["DepPort"]:
            full_route = f"{row1['DepPort']} - {row1['ArrPort']} - {row2['ArrPort']}"
            prob = (row1["Probability"] * row2["Probability"])
            combined_routes.append((full_route, prob*100))  # back to percentage

# Display as DataFrame
combined_df = pd.DataFrame(combined_routes, columns=["Route", "Combined_Percentage"])
full_routes_fr_RU_to_NL.append(combined_df)


sum(RU_hotpotpath_likelyhood["Probability"])


    

# Phase 2
# extract ports in the hotspot list left in the paths from RU port to hotspot
# select only paths from the second port at the hotspot area and their destination
# not in NL
nbs_hotspotPort = path_frRUPort_to_tophotspot['ArrPort'].unique()
mask_hotpot_depPort = paths_fr2ndPort_to_nextPort['DepPort'].isin(nbs_hotspotPort)
path_frHotpot_to_nonNLport = paths_fr2ndPort_to_nextPort[mask_hotpot_depPort]
path_frHotpot_to_nonNLport = path_frHotpot_to_nonNLport[~mask_NLport]

# count and plot pair hotspot-destination
cons_pair_cntry = zip(path_frHotpot_to_nonNLport['Country'], 
                      path_frHotpot_to_nonNLport['Arr_Country'])    
cons_pair_cntry_count = collections.Counter(list(cons_pair_cntry))
cons_pair_cntry_noloop = cons_pair_cntry_count.copy()
# # Filter out rows where ArrPort is 'Russia'
# cond1 = path_frHotpot_to_nonNLport['ArrPort'] == 'Russia'

# # Filter out rows where Country is the same as Arr_Country
# cond2 = path_frHotpot_to_nonNLport['Country'] == path_frHotpot_to_nonNLport['Arr_Country']

# # Combine both conditions
# to_delete = cond1 | cond2

# # Drop the rows that meet either condition
# path_frHotpot_to_nonNLport_noloop  = path_frHotpot_to_nonNLport[~to_delete].reset_index(drop=True)
# # to_delete_key = []
# # for key, value in cons_pair_cntry_count.items():
# #     key1, key2 = key
# #     if key2 =='Russia':
# #         to_delete_key.append(key)
# #     if key1 == key2:
# #         to_delete_key.append(key)
# # for key in to_delete_key:
# #     del cons_pair_cntry_noloop[key]
# country_count = collections.Counter(path_frHotpot_to_nonNLport['Arr_Country'])
# countries_count_noloop = collections.Counter(path_frHotpot_to_nonNLport_noloop['Arr_Country'])
to_RU = {}
cylic_trip = {}
for hotspot in tophotspot:
    for key, value in cons_pair_cntry_count.items():
        key1, key2 = key
        if key2 == 'Russia':
            to_RU[key] = value
        if key1 == key2:
            cylic_trip[key] =  value

# print(sum(to_RU.values()))            
# print(sum(cylic_trip.values()))        
# (sum(cons_pair_cntry_count.values()) - sum(to_RU.values()) - sum(cylic_trip.values()))/ sum(cons_pair_cntry_count.values()) * 100 

# because the function above does not remove arrival port in RU properly, extra removal
mask_RU = path_frHotpot_to_nonNLport['Arr_Country'].isin(['Russia'])
path_frHotpot_to_nonNLenRUport = path_frHotpot_to_nonNLport[~mask_RU]
#* paths from hotspot to its neighbors does not contain Eu, Ru, and loops
path_frHotpot_to_nonNLenRUenselfport = path_frHotpot_to_nonNLenRUport[
    path_frHotpot_to_nonNLenRUport['Country'] != path_frHotpot_to_nonNLenRUport['Arr_Country']
]
hotpotport_to_nextport_count = collections.Counter(path_frHotpot_to_nonNLenRUport['Arr_Country'])
# extract next nbs from pre consecutive ports
next_nbs = path_frHotpot_to_nonNLenRUenselfport['ArrPort'].unique()
#* next path connected to the previous shared port (hotspot)
next_paths = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(next_nbs)]
#* direct path to NL(RU-hotspot-nextstop-RU)
next_paths_dir_to_NL = next_paths[next_paths['Arr_Country'] == 'Netherlands']
next_paths_dir_to_NL_unique = next_paths_dir_to_NL.drop_duplicates(
    subset = ['IMO', 'Country', 'Arr_Country']).reset_index()
# Count and plot pair for 3rd destination vs NL port
cons_pair_next_cntry = zip(next_paths_dir_to_NL['Country'], 
                      next_paths_dir_to_NL['Arr_Country'])

next_country_count = collections.Counter(list(cons_pair_next_cntry))

# IMO for each countries
nr_IMO_each_country = {}
countries = next_paths_dir_to_NL_unique['Country'].unique()
for country in countries:
    nr_IMO_each_country[country] = list(next_paths_dir_to_NL_unique[
        next_paths_dir_to_NL_unique['Country'] == country]['IMO'])
#* determine whether IMO match condition if there are more than 3 stops  
EUpath_to_NL = []
nonEUpath_to_NL = []          
for row in range(len(next_paths_dir_to_NL_unique)):
    path = next_paths_dir_to_NL_unique.iloc[row]
    if path['DepPort'] in eu_ports:
        IMO_per_country = path_frHotpot_to_nonNLenRUenselfport[
            path_frHotpot_to_nonNLenRUenselfport['ArrPort'] == path['DepPort']]['IMO']
        match_depport_path = path_frHotpot_to_nonNLenRUenselfport[
            path_frHotpot_to_nonNLenRUenselfport['ArrPort'] == path['DepPort']]

        
        if path['IMO'] in set(IMO_per_country):
            path_list = match_depport_path[(match_depport_path['IMO'] == path['IMO'])]

            if len(path_list) ==1:
                path_df = path.to_frame().T
                path_combined = pd.concat([path_list, path_df])
                EUpath_to_NL.append(path_combined)
            else:
                time_collect = []
                for row_dep_port in range(len(path_list)):
                    # departure time of an IMO
                    pre_date = path_list.iloc[row_dep_port,
                                                      path_list.columns.get_loc('ArrDate')]
                    cons_date = path_df.iloc[:,path_df.columns.get_loc('DepDate')]
                    cons_date = pd.to_datetime(cons_date)

           
                    # time different between IMO from RU and IMO availabe at its nb
                    time_gap = cons_date - pre_date
                    time_gap_hr = (time_gap.dt.total_seconds() / 3600).iloc[0]
                    time_collect.append(time_gap_hr)
                
                # Filter only positive values and get their indices
                positive_indices = [(i, val) for i, val in enumerate(time_collect) if val > 0]
                
                # Find the one with the minimum value
                if len(positive_indices) == 0:
                    next
                else:
                    min_pos_index, min_pos_value = min(positive_indices, key=lambda x: x[1])
                    path_combined = pd.concat([path_list.iloc[[min_pos_index]], path_df])
                    if path_list.iloc[min_pos_index]['IMO'] in  path_df['IMO'].values:
                        EUpath_to_NL.append(path_combined)
                    else:
                        next

    else:
        nonEUpath_to_NL.append(path.to_frame().T)
EUpath_to_NL = pd.concat(EUpath_to_NL,ignore_index=True)
nonEUpath_to_NL = pd.concat(nonEUpath_to_NL,ignore_index=True)
count_country_EUpath_to_NL = collections.Counter(EUpath_to_NL['Country'])
count_country_nonEUpath_to_NL = collections.Counter(nonEUpath_to_NL['Country'])
# remove df with shiptypes as follow
remove_shiptype = ['Asphalt/Bitumen Tanker', 'Oil Bunkering Tanker', 'Shuttle Tanker']
EUpath_to_NL_filtered = [df for df in EUpath_to_NL if df['ShipType'].unique() not in remove_shiptype]

# trace back to calculate probabilty
path_frRUPort_to_tophotspot

# select the most popular paths
path_hotspot_EUport = [df.iloc[[0]] for df in EUpath_to_NL_filtered]
path_of_part_EUport_to_NL = [df.iloc[[-1]] for df in EUpath_to_NL_filtered]
depPort_at_NL = set(paths_fr2ndhotspotPort_to_NL['Country'])
full_routes_fr_RU_to_NL = []

# Convert to DataFrame for better viewing or plotting
paths_freu1_toNL_likelyhood = pr.path_likelyhood(pd.concat(path_of_part_EUport_to_NL,
                                                           ignore_index= True))
path_of_part_EUport_to_NL_likelyhood = pr.path_likelyhood(pd.concat(path_hotspot_EUport,
                                                           ignore_index= True))
path_hotspot_EUport = pd.concat(path_hotspot_EUport, ignore_index= True)
port_at_B = set(path_hotspot_EUport['DepPort'])


paths_frA_toB = path_frRUPort_to_tophotspot[path_frRUPort_to_tophotspot['ArrPort'].isin(port_at_B)]
RU_hotpotpath_likelyhood = pr.path_likelyhood(paths_frA_toB)
    # Normalize percentages to probabilities
paths_freu1_toNL_likelyhood["Probability"] = paths_freu1_toNL_likelyhood["Percentage"] / 100
path_of_part_EUport_to_NL_likelyhood["Probability"] = path_of_part_EUport_to_NL_likelyhood["Percentage"] / 100
RU_hotpotpath_likelyhood["Probability"] = RU_hotpotpath_likelyhood["Percentage"] / 100
# Combine segments where seg1.ArrPort == seg2.DepPort
combined_routes = []
    
for i, row1 in RU_hotpotpath_likelyhood.iterrows():
    for j, row2 in path_of_part_EUport_to_NL_likelyhood.iterrows():
        for k, row3 in paths_freu1_toNL_likelyhood.iterrows():
            if (row1["ArrPort"] == row2["DepPort"]) and row2["ArrPort"] == row3["DepPort"]:
                full_route = f"{row1['DepPort']} - {row1['ArrPort']} - {row2['ArrPort']} - {row3['ArrPort']}"
                prob = row1["Probability"] * row2["Probability"] * row3['Probability']
                combined_routes.append((full_route, prob * 100))  # back to percentage
    
#* Display as DataFrame
combined_df = pd.DataFrame(combined_routes, columns=["Route", "Combined_Percentage"])
full_routes_fr_RU_to_NL.append(combined_df)
combined_2stop_routes = combined_df
#* for non EU path
nonEUpath_to_NL = pd.concat(nonEUpath_to_NL, ignore_index=True)
port_nonEU_v1 = nonEUpath_to_NL['DepPort'].unique()
path_frHotpot_to_noneuport_v1 = path_frHotpot_to_nonNLenRUenselfport[
    path_frHotpot_to_nonNLenRUenselfport['ArrPort'].isin(port_nonEU_v1)]
# calculate likelyhood
nonEUpath_to_NL_likelyhood = pr.path_likelyhood(nonEUpath_to_NL)
path_frHotpot_to_noneuport_v1_likelyhood = pr.path_likelyhood(path_frHotpot_to_noneuport_v1)
    # Normalize percentages to probabilities
nonEUpath_to_NL_likelyhood["Probability"] = nonEUpath_to_NL_likelyhood["Percentage"] / 100
path_frHotpot_to_noneuport_v1_likelyhood["Probability"] = path_frHotpot_to_noneuport_v1_likelyhood[
    "Percentage"] / 100
# Combine segments where seg1.ArrPort == seg2.DepPort
full_routes_fr_RU_to_NL = []
combined_routes = []
    
for i, row1 in RU_hotpotpath_likelyhood.iterrows():
    for j, row2 in path_frHotpot_to_noneuport_v1_likelyhood.iterrows():
        for k, row3 in nonEUpath_to_NL_likelyhood.iterrows():
            if (row1["ArrPort"] == row2["DepPort"]) and row2["ArrPort"] == row3["DepPort"]:
                full_route = f"{row1['DepPort']} - {row1['ArrPort']} - {row2['ArrPort']} - {row3['ArrPort']}"
                prob = row1["Probability"] * row2["Probability"] * row3['Probability']
                combined_routes.append((full_route, prob * 100))  # back to percentage

#* Display as DataFrame
combined_df = pd.DataFrame(combined_routes, columns=["Route", "Combined_Percentage"]).sort_values(by = 'Combined_Percentage')
full_routes_fr_RU_to_NL.append(combined_df)

sum(nonEUpath_to_NL_likelyhood['Probability'])
sum(combined_2stop_routes['Combined_Percentage'])
a = next_paths_dir_to_NL_unique[next_paths_dir_to_NL_unique['IMO'] == 9292046]
a = alltankers_adjusted[alltankers_adjusted['IMO'] == 9370848]

aaa = []
aaa =+ 2
aaa =+ 3
             
a = path_frHotpot_to_nonNLport[path_frHotpot_to_nonNLport['ArrPort'] == 'Sines']
path_frHotpot_to_nonNLport

a = alltankers_adjusted[alltankers_adjusted['DepPort'] == 'Pengerang Terminal']
9350654

# find IMO both in RU and NL
IMO_uniq = paths_fr2ndPort_to_NL['IMO'].unique()
IMO_in_both_RU_NL = []
IMO_in_both_RU_NL = [
    imo for imo in IMO_uniq
    if set(alltankers_adjusted[alltankers_adjusted['IMO'] == imo]['Country']) >= {'Netherlands', 'Russia'}
]


# %% Analyse
all_paths_freq_RU_to_nonEUport = pr.path_freq_of_a_IMO_fr_A_to_B(
    count_IMO_frRUPort_to_nonEUenRUnbs, 
    alltankers_adjusted)
all_paths_freq_RU_to_nonEUport = all_paths_freq_RU_to_nonEUport[
    all_paths_freq_RU_to_nonEUport['Sum'] >0]
total_fred_route_percntry = pd.Series(all_paths_freq_RU_to_nonEUport.sum())

all_paths_2ndhotspot_to_NL = pr.path_freq_of_a_IMO_fr_B_to_NL(
    count_IMO_fr2ndPort_to_NL, alltankers_adjusted)
all_paths_2ndhotspot_to_NL = all_paths_2ndhotspot_to_NL[
    all_paths_2ndhotspot_to_NL['Sum'] >0]
total_fred_route_toNL = all_paths_2ndhotspot_to_NL.sum()


# percentage of IMO that revisit hotspots from RU
perc_IMO_revisit_hotspot_allIMO = (len(all_paths_freq_RU_to_nonEUport[
    all_paths_freq_RU_to_nonEUport['Sum'] >= 2])/len(all_paths_freq_RU_to_nonEUport))*100
path_frRUPort_to_nonEUenRUnbs = path_frRUPort_to_nonEUenRUnbs.merge(port_itscountry,
                                                                    left_on = ['ArrPort'],
                                                                    right_on = ['DepPort'],
                                                                    
                                                                    )
all_shiptypes = sorted(
    set(path_frRUPort_to_nonEUenRUnbs['ShipType'].unique()) |
    set(paths_fr2ndPort_to_NL['ShipType'].unique())
)
# Define consistent colors using seaborn or matplotlib
color_palette = sns.color_palette("tab10", n_colors=len(all_shiptypes))
color_map = dict(zip(all_shiptypes, color_palette))

# Reindex columns to ensure consistent order
path_frRUPort_to_nonEUenRUnbs_gr = path_frRUPort_to_nonEUenRUnbs.groupby('Country_y')['ShipType'].value_counts().unstack(fill_value=0)
paths_fr2ndPort_to_NL_gr = paths_fr2ndPort_to_NL.groupby('Country')['ShipType'].value_counts().unstack(fill_value=0)
path_frRUPort_to_nonEUenRUnbs_gr = path_frRUPort_to_nonEUenRUnbs_gr.reindex(columns=all_shiptypes, fill_value=0)
paths_fr2ndPort_to_NL_gr = paths_fr2ndPort_to_NL_gr.reindex(columns=all_shiptypes, fill_value=0)

ax = path_frRUPort_to_nonEUenRUnbs_gr.plot.bar(stacked=True, color=[color_map[st] for st in path_frRUPort_to_nonEUenRUnbs_gr.columns])
plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
plt.title('Contribution of different shiptypes in oil transportation from RU to the second countries')
ax.set_xlabel('Arrival Countries')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=90)


ax = paths_fr2ndPort_to_NL_gr.plot.bar(stacked=True, color=[color_map[st] for st in paths_fr2ndPort_to_NL_gr.columns])
# plt.legend(
#             bbox_to_anchor=(0.5, -0.6),
#             loc="lower center",
#             borderaxespad=0,
#             frameon=False,
#             ncol=3,
#         )
plt.title('Contribution of different shiptypes in oil transportation from snd port onwards')
ax.set_xlabel('Arrival Countries')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=90)
start_IMO = {} # start from a specific RU port. Expect a dict of dict. with the
# first layer contain RU port names and its connected IMO info in general.
# The secondlayer key: nb name and its attributes
cons_IMO = {} # the next second port met conditions (no EU port) and contain a connected IMO
cons_IMO_nr = [] # contain a connected IMO
# loop through all neighbours of Novorossiysk
track_route_fr_RU_to_2ndPort_and_connected_IMO = []
for edge in edges_Nov:
    # extract route from a RU port to its neighbout
    start_RU_port = edge[0]
    route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([start_RU_port]) & 
                        alltankers_adjusted['ArrPort'].isin([edge[1]])]
    arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
    # extract all IMO available at the arrival port of the first trip from RU
    diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(arr_port)]
    # for each nb, calculate time gap between IMO from RU to 2nd port and 
    # the IMO available at the 2nd port
    
    for row in range(len(route_from_RUport_to_its_nb)):
        
        # arrive time of IMO travel from RU to its nb
        arr_time_of_IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row,
                                              route_from_RUport_to_its_nb.columns.get_loc('ArrDate')]
        IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row:row+1]
        test_test_ = pd.DataFrame(columns = IMO_fr_RU_to_2ndPort.columns)
        # loop through all IMO availabel in a nb
        for row_dep_port in range(len(diff_IMO_at_2ndPort)):
            # departure time of an IMO
            dep_time_of_IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port,
                                              diff_IMO_at_2ndPort.columns.get_loc('DepDate')]
            dep_port_per_row = diff_IMO_at_2ndPort.iloc[row_dep_port,
                                              diff_IMO_at_2ndPort.columns.get_loc('DepPort')]
            IMO_avai_at_2ndPort = diff_IMO_at_2ndPort.iloc[row_dep_port:
                                              row_dep_port+1]
    
            # time different between IMO from RU and IMO availabe at its nb
            time_gap = dep_time_of_IMO_avai_at_2ndPort - arr_time_of_IMO_fr_RU_to_2ndPort
            time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
            # base on threshold assump to take for oil transshipment. if met 
            # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
            if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= low_t_time) & (abs(time_gap_hr) < up_t_time)):
                #print(f'time gap after the condition {time_gap_hr}')
    
    
                test_test_ = pd.concat([IMO_fr_RU_to_2ndPort, IMO_avai_at_2ndPort])
                if len(test_test_) != 0:
                    
                    track_route_fr_RU_to_2ndPort_and_connected_IMO.append(test_test_)
                index = 1
                # check whether the key(nb name) in the dict, not create new. 
                # Otherwise, append 
                if dep_port_per_row not in list(cons_IMO.keys()):
                    cons_IMO[dep_port_per_row] =  tuple([[
                        index, {'DepPort' : diff_IMO_at_2ndPort['DepPort'].iloc[row_dep_port],
                                'ArrPort' : diff_IMO_at_2ndPort['ArrPort'].iloc[row_dep_port],
                                'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                      'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                      'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                    IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                    cons_IMO_nr.append(IMO)
                else:
                    index = len(cons_IMO[dep_port_per_row]) + 1  
                    cons_IMO[dep_port_per_row] += tuple([[
                        index, {'DepPort' : diff_IMO_at_2ndPort['DepPort'].iloc[row_dep_port],
                                'ArrPort' : diff_IMO_at_2ndPort['ArrPort'].iloc[row_dep_port],
                                'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                      'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                      'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                    IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                    cons_IMO_nr.append(IMO)
            else:
                next
            # add main key of the dictionary, the key is a RU port name
    start_IMO[start_RU_port] = cons_IMO
IMO_RU_nonEUport = paths_fr2ndPort_to_NL['IMO'].unique()
IMO_2ndport_NL = path_frRUPort_to_nonEUenRUnbs['IMO'].unique()
IMO_intersection = np.intersect1d(IMO_2ndport_NL, IMO_RU_nonEUport)

# Seasonality
months_collection_RU = []
for imo in list(all_paths_freq_RU_to_nonEUport['IMO']):
    imo_seq = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    imo_seq_RU = imo_seq[imo_seq['Country'] == 'Russia']
    imo_seq_RU_date = set(imo_seq_RU['DepDate'].dt.month)
    months_collection_RU += list(imo_seq_RU_date) 
months_collection_NL = []
for imo in list(all_paths_2ndhotspot_to_NL['IMO']):
    imo_seq = alltankers_adjusted[alltankers_adjusted['IMO'] == imo]
    imo_seq_NL = imo_seq[imo_seq['Country'] == 'Netherlands']
    imo_seq_NL_date = set(imo_seq_NL['DepDate'].dt.month)
    months_collection_NL += list(imo_seq_NL_date) 
# Convert to pandas Series and count occurrences
month_RU_counts = pd.Series(months_collection_RU).value_counts().sort_index()
month_NL_counts = pd.Series(months_collection_NL).value_counts().sort_index()
# %% Visualization
# Combine all unique ShipTypes from both DataFrames
# 1. Combine all shiptypes
all_shiptypes = sorted(
    set(path_frRUPort_to_nonEUenRUnbs['ShipType'].unique()) |
    set(paths_fr2ndPort_to_nextPort['ShipType'].unique())
)

# 2. Assign consistent colors
color_palette = sns.color_palette("tab10", n_colors=len(all_shiptypes))
color_map = dict(zip(all_shiptypes, color_palette))

# 3. Group data and align columns
st_gr = path_frRUPort_to_nonEUenRUnbs.groupby('Arr_Country')[
    'ShipType'].value_counts().unstack(fill_value=0)
snd_gr = paths_fr2ndPort_to_nextPort.groupby(
    'Country')['ShipType'].value_counts().unstack(fill_value=0)

st_gr = st_gr.reindex(columns=all_shiptypes, fill_value=0)
snd_gr = snd_gr.reindex(columns=all_shiptypes, fill_value=0)
# Sum across ship types and select top 20 countries (rows)
top_20_countries_st = st_gr.sum(axis=1).sort_values(ascending=False).head(20).index
top_20_countries_snd = snd_gr.sum(axis=1).sort_values(ascending=False).head(20).index
# Filter both dataframes to keep only top 20 countries
st_gr_top20 = st_gr.loc[top_20_countries_st]
snd_gr_top20 = snd_gr.loc[top_20_countries_snd]


ax = st_gr_top20.plot.bar(stacked=True, color=[color_map[st] for st in st_gr_top20.columns])
plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
plt.title('Contribution of different shiptypes in oil transportation from RU to the top 20 second countries')
ax.set_xlabel('Arrival Countries')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=90)


ax = snd_gr_top20.plot.bar(stacked=True, color=[color_map[st] for st in snd_gr_top20.columns])
plt.legend(
            bbox_to_anchor=(0.5, -0.6),
            loc="lower center",
            borderaxespad=0,
            frameon=False,
            ncol=3,
        )
plt.title('Contribution of different shiptypes in oil transportation from the top 20 second countries port onwards')
ax.set_xlabel('Arrival Countries')
ax.set_ylabel('Nr. of vessels')
ax.tick_params(axis='x', labelrotation=90)


# Convert the two Series to DataFrames (removing 'IMO' and 'Sum' if included)
stage1 = total_fred_route_percntry.drop(['IMO', 'Sum'])
stage2 = total_fred_route_toNL.drop(['IMO', 'Sum'])

# Rename index for consistency (remove "_NL" and match with stage1)
stage2.index = [key.replace("_NL", "") for key in stage2.index]
stage1.index = [key.replace("RU_", "") for key in stage1.index]
# Combine into a DataFrame
combined_df = pd.DataFrame({
    'RU to 2nd Port': stage1,
    '2nd Port to NL': stage2
}).fillna(0)  # fill missing with 0
# Plotting
ax = combined_df.plot.bar(rot=90, width=0.7)

# Formatting
plt.title("Vessel Flow: From Russia to Non-EU Ports and Then to NL")
plt.ylabel("Number of port calls")
plt.xlabel("Country (2nd Port)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Extracting specific keys from dictionary
count_port_RUpathHotspot_match_toNL = {key: count_port_path_frRUPort_to_hotspotport[key]
                                       for key in count_port_path_frRUPort_to_hotspotport.keys() 
                                       & count_port_path_2ndPort_t0_NL.keys()}
count_port_RUpathHotspot_match_toNL_df = pd.DataFrame.from_dict(
    count_port_RUpathHotspot_match_toNL, orient = 'index',
    columns = ['Count']).reset_index()

count_port_path_2ndPort_t0_NL_df = pd.DataFrame.from_dict(
    count_port_path_2ndPort_t0_NL, orient = 'index',
    columns = ['Count']).reset_index()
combine_RU_and_NL_port_count = pd.merge(count_port_RUpathHotspot_match_toNL_df,
                                        count_port_path_2ndPort_t0_NL_df, on='index')
combine_RU_and_NL_port_count.columns = ['Ports', 'Nr. arrival ports from RU', 'Nr. dep ports to NL']
ax = combine_RU_and_NL_port_count.plot.bar(x = 'Ports', rot=90, width=0.7)

# Formatting
plt.title("Shared Ports relations between RU and NL")
plt.ylabel("Number of port calls")
plt.xlabel("Country (Shared Ports)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# plot path frequency of hotspot-nextstop vs nextstop-NL
# convert to dataframe

# Create a new dict with only the first country as key
# Prepare and filter the top 35 entries
hotpotport_to_nextport_count_df = (
    pd.DataFrame(list(hotpotport_to_nextport_count.items()), columns=['Country', 'Count'])
    .set_index('Country')
    .sort_values(by='Count')
    .tail(35)  # Take top 35
)

next_country_count_cleaned = {k[0]: v for k, v in next_country_count.items()}
next_country_count_df = (
    pd.DataFrame(list(next_country_count_cleaned.items()), columns=['Country', 'Count'])
    .set_index('Country')
    .sort_values(by='Count')
    .tail(35)  # Take top 35
)

# Create figure and subplots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))  # Two subplots side by side

# Plot on each subplot
hotpotport_to_nextport_count_df.plot.bar(ax=ax[0], rot=90, width=0.7, legend=False)
next_country_count_df.plot.bar(ax=ax[1], rot=90, width=0.7, legend=False)

# Set titles and labels for each subplot
ax[0].set_title("HotspotPort → NextPort")
ax[0].set_ylabel("Number of Port Calls")
ax[0].set_xlabel("Country")

ax[1].set_title("NextPort → NL")
ax[1].set_ylabel("Number of Port Calls")
ax[1].set_xlabel("Country")

plt.tight_layout()
plt.show()

# plot the frequency vessels leave each port in RU
count_RUport_df = pd.DataFrame.from_dict(count_RUport, orient = 'index', 
                                         columns = ['Count'])
ax = count_RUport_df.plot.bar(rot=90, width=0.7)

# Formatting
plt.title("Frequency ports used")
plt.ylabel("Number of port calls")
plt.xlabel("Port names")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# plot seasonality
# Plotting

# Create figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))  # Default: sharey=False

# Plot for Russia
axes[0].plot(month_RU_counts.index, month_RU_counts.values, marker='o', linestyle='-', color='tab:blue')
axes[0].set_title("Active IMOs per Month – Russia")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Number of IMO")
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
axes[0].grid(True)

# Plot for Netherlands
axes[1].plot(month_NL_counts.index, month_NL_counts.values, marker='o', linestyle='-', color='tab:green')
axes[1].set_title("Active IMOs per Month – Netherlands")
axes[1].set_xlabel("Month")
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
axes[1].grid(True)

# Set overall layout
plt.suptitle("Monthly IMO Activity – Russia vs Netherlands", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # reserve space for suptitle
plt.show()