# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:43:16 2025

@author: Duyen
"""

# %% Load in library

import pandas as pd
import os
os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Code import data_preprocessing as pp

import re
import csv
import networkx as nx
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from multiprocessing import Pool, cpu_count
import unicodedata

from itertools import islice  
import itertools
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# %% 
# %% 

os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
# Create directory to story data  
datapath_nested_directory = './preprocessing/pp_inter_ouput' 
try: 
    os.makedirs(datapath_nested_directory)
    print(f"Nested directory' '{datapath_nested_directory}' created successfully")
except FileExistsError:
    print(f"One or more direcotries in '{datapath_nested_directory}' aldready exist")
    
except PermissionError():
    print(f"Permission denied: Unable to create '{datapath_nested_directory}")
except Exception as e:
    print(f"An error occured: {e}")
# import data
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

# create network
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
## direct-graph
direct_graph = nx.DiGraph()
direct_graph.add_edges_from(network_edges)
# Create all combination of RU and NL ports
comb_ru_nl_ports = list(itertools.product(port_of_russia, NL_ports))


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
up_t_time = 48
low_t_time = 2
scnd_in_day = 1*24*60*60
# extract 1 hop
# aim of this task is to find a connection trip at the nb port of the first trip
# extract edges from st.Peterburg and Novorossiysk
# neighbors_Nov = list(Graph_whole_dataset.neighbors('Novorossiysk'))
# filter_neighbors_Nov = [n for n in neighbors_Nov if n not in eu_ports]
# out_edge_node_neighbour = []
# extract only neighbours that belongs to the routes going from Novorossiysk 
edges_Nov = list(set(list(Graph_whole_dataset.out_edges('Novorossiysk',))))
start_IMO = {} # start from a specific RU port. Expect a dict of dict. with the
# first layer contain RU port names and its connected IMO info in general.
# The secondlayer key: nb name and its attributes
cons_IMO = {} # the next second port met conditions (no EU port) and contain a connected IMO
cons_IMO_nr = [] # contain a connected IMO
# loop through all neighbours of Novorossiysk
track_route_fr_RU_to_2ndPort_and_connected_IMO = []
for edge in edges_Nov:
    # extract route from a RU port to its neighbout
    route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([edge[0]]) & 
                        alltankers_adjusted['ArrPort'].isin([edge[1]])]
    # extract all IMO available at the arrival port of the first trip from RU
    diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([edge[1]])]
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
                if edge[1] not in list(cons_IMO.keys()):
                    cons_IMO[edge[1]] =  tuple([[index, {'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                     'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                     'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                    IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                    cons_IMO_nr.append(IMO)
                else:
                    index = len(cons_IMO[edge[1]]) + 1  
                    cons_IMO[edge[1]] += tuple([[index, {'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                     'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                     'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                    IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                    cons_IMO_nr.append(IMO)
            else:
                next

    # add main key of the dictionary, the key is a RU port name
    start_IMO[edge[0]] = cons_IMO
    
 
# check we    
# unique set of connected IMO considering from all nbs
cons_IMO_nr_uniq = set(cons_IMO_nr)
# identify which connected IMO were to the NL
cons_IMO_were_to_NL = cons_IMO_nr_uniq.intersection(IMO_in_NL['IMO'].unique())
# loop through a dict of connected IMO at nb nodes of the first RU trip. Only 
# select connected IMO that were to NL
snd_IMO_were_to_NL_df = pd.DataFrame()

for key, value in cons_IMO.items():

            for ind, timestamp in value:
                IMO_snd_port = timestamp['IMO']
                if IMO_snd_port in cons_IMO_were_to_NL:
                    each_snd_IMO_were_to_NL_df = pd.DataFrame([[IMO_snd_port,key,
                                                                timestamp['DepDate'],
                                                                timestamp['ArrDate']                                                      
                                                            ]])
                    
                    snd_IMO_were_to_NL_df = pd.concat([snd_IMO_were_to_NL_df, each_snd_IMO_were_to_NL_df])
snd_IMO_were_to_NL_df.columns = ['IMO', 'DepPort','DepDate', 'ArrDate']
# remove IMO or 2nd ports in Eu or RU ports
snd_IMO_were_to_NL_df = snd_IMO_were_to_NL_df[~snd_IMO_were_to_NL_df['DepPort'].isin(
    eu_ports) & ~snd_IMO_were_to_NL_df['DepPort'].isin(port_of_russia) ] #add columns-depPort
# write a function that select all trips of connected IMO from 2nd port (nbs)
# to NL
# a df contain potential IMOs and its sequence travel from RU to NL port
pot_imo_from_RU_to_NL = pd.DataFrame(columns = snd_IMO_were_to_NL_df.columns)
for row in range(len(snd_IMO_were_to_NL_df)):
    imo_from_2ndport_to_NL = alltankers_adjusted[
        alltankers_adjusted['IMO'] == snd_IMO_were_to_NL_df['IMO'].iloc[row]]
    time_imo_at_NL_ports = imo_from_2ndport_to_NL[imo_from_2ndport_to_NL['DepPort'].isin(NL_ports)]
    time_imo_at_NL_ports = time_imo_at_NL_ports['ArrDate']
    if any(snd_IMO_were_to_NL_df['DepDate'].iloc[row] < time_imo_at_NL_ports):
        pot_imo_from_RU_to_NL = pd.concat([pot_imo_from_RU_to_NL,
                                           snd_IMO_were_to_NL_df.iloc[row: row+1]])
    else:
        next
# extract only routes that directly travek from 2nd port to NL or
# stop by more ports before arriving in NL but no more oil transhipment-the 
# same ship from 2nd port to the Netherlands
# A condition for a ship that stop at more than one ports, should not visit RU 
# again or visit a port included in the route more than twice

# select one hop
# select more hops but check conditions
# check in the dataframe from the timestamp at the 2nd ports to the time it
# reaches the Netherlands for the first time 

# filter route from 2nd port to NL, spatial consistency (not go back to RU or
# revisit a node twice) and temporal consistency ( time arrive in NL is later
# than time leave the second port)
one_oil_trans_fr_RU_to_NL = []
for row in range(len(pot_imo_from_RU_to_NL)):
    # extract only sequence of a potential IMO
    pot_imo = alltankers_adjusted[alltankers_adjusted['IMO'] ==
                                  pot_imo_from_RU_to_NL['IMO'].iloc[row]].reset_index(drop = True)
    # locate index of oil transshipment at the 2nd port based on deptime, 
    # depport, arridate
    snd_target_port = pot_imo[(pot_imo['DepPort'] == pot_imo_from_RU_to_NL['DepPort'].iloc[row])
                                        & (pot_imo['DepDate'] == pot_imo_from_RU_to_NL['DepDate'].iloc[row])
                                        & (pot_imo['ArrDate'] == pot_imo_from_RU_to_NL['ArrDate'].iloc[row])]
    # if it is a direct route to NL save, else only extract trip segment from
    # the moment of potential oil transshipment to the first time it arrives
    # in the NL
    one_oil_trans = pd.DataFrame(columns = alltankers_adjusted.columns)
    if snd_target_port['ArrPort'].isin(NL_ports).any():
        one_oil_trans = pd.concat([one_oil_trans, snd_target_port])
    else:
        row_nr_of_target_2nd_port = snd_target_port.index[0]
        
        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:len(pot_imo)]
        first_nl_port = pot_imo_from_2nd_target_port[
            pot_imo_from_2nd_target_port['DepPort'].isin(NL_ports)].index[0]
        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:first_nl_port]
        if ~pot_imo_from_2nd_target_port['DepPort'].isin(port_of_russia).any():
            # create network
            edges = []
            for n in range(len(pot_imo_from_2nd_target_port)):
                info = tuple([pot_imo_from_2nd_target_port['DepPort'].iloc[n], 
                              pot_imo_from_2nd_target_port['ArrPort'].iloc[n]])
                edges.append(info)
            # create graph
            ## multi-direct-graph
            Graph_seg_route = nx.MultiDiGraph()
            Graph_seg_route.add_edges_from(edges)
            # remove loop
            Graph_seg_route.remove_edges_from(list(nx.selfloop_edges(Graph_seg_route)))
            nodes_degree = list(Graph_seg_route.degree())
            # check degree for each nodes in the trip segment
            node_with_more_2_degree = [node for node, value in nodes_degree if value >=3]
            # if there are nodes with > 2 degree, the route is invalid. Otherwise, append the route
            if len(node_with_more_2_degree) == 0:
                one_oil_trans = pd.concat([one_oil_trans,
                                                       pot_imo_from_2nd_target_port])
    if len(one_oil_trans) != 0:
        one_oil_trans_fr_RU_to_NL.append(one_oil_trans)
                
temp_one_oil_trans_fr_RU_to_NL = pd.concat(one_oil_trans_fr_RU_to_NL, ignore_index=True)
# retrieve route from RU to the 2nd port and make a connection with the selected IMO
# available at the second port
route_fr_RU_2ndPort_fn = []

imo_unique_fr_RU_to_NL = temp_one_oil_trans_fr_RU_to_NL['IMO'].unique()


for i in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
    seg_df = track_route_fr_RU_to_2ndPort_and_connected_IMO[i]
    if seg_df['IMO'].iloc[1] in imo_unique_fr_RU_to_NL:
        route_fr_RU_2ndPort_fn.append(seg_df)


# Final combined list
combined_routes = []

for df2 in route_fr_RU_2ndPort_fn:
    for df1 in one_oil_trans_fr_RU_to_NL:
        if df2.iloc[-1].equals(df1.iloc[0]):
            # Drop first row of df1 to avoid duplicate
            merged = pd.concat([df2, df1.iloc[1:]], ignore_index=True)
            combined_routes.append(merged)
            
            
# CODE 15_7
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:41:10 2025

@author: Duyen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:43:16 2025

@author: Duyen
"""

# %% Load in library

import pandas as pd
import os
os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Code import data_preprocessing as pp
from Code import data_processing as pr
import re
import csv
import networkx as nx
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from multiprocessing import Pool, cpu_count
import unicodedata

from itertools import islice  
import itertools
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# %% 
# %% 

os.chdir("D:/Dropbox/Duyen/University/Master/Year 2/Internship")
# Create directory to story data  
datapath_nested_directory = './preprocessing/pp_inter_ouput' 
try: 
    os.makedirs(datapath_nested_directory)
    print(f"Nested directory' '{datapath_nested_directory}' created successfully")
except FileExistsError:
    print(f"One or more direcotries in '{datapath_nested_directory}' aldready exist")
    
except PermissionError():
    print(f"Permission denied: Unable to create '{datapath_nested_directory}")
except Exception as e:
    print(f"An error occured: {e}")
# import data
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

# create network
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
## direct-graph
direct_graph = nx.DiGraph()
direct_graph.add_edges_from(network_edges)
# Create all combination of RU and NL ports
comb_ru_nl_ports = list(itertools.product(port_of_russia, NL_ports))


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
up_t_time = 48
low_t_time = 2
scnd_in_day = 1*24*60*60
# extract 1 hop
# aim of this task is to find a connection trip at the nb port of the first trip
# extract edges from st.Peterburg and Novorossiysk
# neighbors_Nov = list(Graph_whole_dataset.neighbors('Novorossiysk'))
# filter_neighbors_Nov = [n for n in neighbors_Nov if n not in eu_ports]
# out_edge_node_neighbour = []
# extract only neighbours that belongs to the routes going from Novorossiysk 
m = 2
n = 0
edges_Nov = list(set(list(Graph_whole_dataset.out_edges('Novorossiysk',))))
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
    
    track_route_fr_RU_to_2ndPort_and_connected_IMO, start_IMO, cons_IMO, cons_IMO_nr = pr.potential_IMO_at_shared_port(
        track_route_fr_RU_to_2ndPort_and_connected_IMO, 
                                     route_from_RUport_to_its_nb, start_RU_port, 
                                     diff_IMO_at_2ndPort, 
                                     scnd_in_day, low_t_time, up_t_time )
n = n+1
while n<m:
   for node1, value1 in start_IMO.items():
       to_delete = []
       for node2, value2 in value1.items():
           if (node2 in port_of_russia) | (node2 in eu_ports):
               to_delete.append(node2)
       for key in to_delete:
           del value1[key]

   tracking_snd_oiltransshipment_imo_list = []
   for row in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
       each_nb_route = track_route_fr_RU_to_2ndPort_and_connected_IMO[row]
       arr_pre_port = each_nb_route.iloc[len(each_nb_route)-2]['ArrPort']
       arr_next_port = each_nb_route.iloc[
           len(each_nb_route)-1]['ArrPort']
       if (arr_pre_port not in eu_ports) and (arr_pre_port not in port_of_russia):
           if (arr_next_port not in eu_ports) and (arr_next_port not in port_of_russia):
               tracking_snd_oiltransshipment_imo_list.append(
                   track_route_fr_RU_to_2ndPort_and_connected_IMO[row])
               
               
   # just for now, with IMO that has a selfloop will be deleted. Should be in the preprocessing by combining the selfloop..
   tracking_filtered_snd_oiltransshipment_imo_list = []
   for row in range(len(tracking_snd_oiltransshipment_imo_list)):
       each_nb_route = tracking_snd_oiltransshipment_imo_list[row]
       if each_nb_route.iloc[len(each_nb_route)-1]['ArrPort'] != each_nb_route.iloc[len(each_nb_route)-1]['DepPort']:
           tracking_filtered_snd_oiltransshipment_imo_list.append(
              each_nb_route)
           

   for row_in_imo_list in range(len(tracking_filtered_snd_oiltransshipment_imo_list)): #tracking_snd_oiltransshipment_imo_list -has more row and each row is a different label
       

       # loop through all neighbours of Novorossiysk
       

       # extract route from a RU port to its neighbout
       route_from_RUport_to_its_nb = tracking_filtered_snd_oiltransshipment_imo_list[row_in_imo_list].iloc[1:2]
       arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
       # extract all IMO available at the arrival port of the first trip from RU
       diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(arr_port)]
       # for each nb, calculate time gap between IMO from RU to 2nd port and 
       # the IMO available at the 2nd port
       track_route_fr_RU_to_2ndPort_and_connected_IMO_v1, start_IMO_v1, cons_IMO_v1, cons_IMO_nr_v1 = pr.potential_IMO_at_shared_port(
           
                                        route_from_RUport_to_its_nb, start_RU_port, 
                                        diff_IMO_at_2ndPort, 
                                        scnd_in_day, low_t_time, up_t_time)
       
track_route_fr_RU_to_2ndPort_and_connected_IMO_v1       



    
# Phase 2   
# unique set of connected IMO considering from all nbs
cons_IMO_nr_uniq = set(cons_IMO_nr)
# identify which connected IMO were to the NL
cons_IMO_were_to_NL = cons_IMO_nr_uniq.intersection(IMO_in_NL['IMO'].unique())
# loop through a dict of connected IMO at nb nodes of the first RU trip. Only 
# select connected IMO that were to NL
snd_IMO_were_to_NL_df = pd.DataFrame()

for key, value in cons_IMO.items():


            for ind, timestamp in value:
                IMO_snd_port = timestamp['IMO']
                if IMO_snd_port in cons_IMO_were_to_NL:
                    each_snd_IMO_were_to_NL_df = pd.DataFrame([[IMO_snd_port,key,
                                                                timestamp['DepDate'],
                                                                timestamp['ArrDate']                                                      
                                                            ]])
                    
                    snd_IMO_were_to_NL_df = pd.concat([snd_IMO_were_to_NL_df, each_snd_IMO_were_to_NL_df])
snd_IMO_were_to_NL_df.columns = ['IMO', 'DepPort','DepDate', 'ArrDate']
# remove IMO or 2nd ports in Eu or RU ports
snd_IMO_were_to_NL_df = snd_IMO_were_to_NL_df[~snd_IMO_were_to_NL_df['DepPort'].isin(
    eu_ports) & ~snd_IMO_were_to_NL_df['DepPort'].isin(port_of_russia) ] #add columns-depPort
# write a function that select all trips of connected IMO from 2nd port (nbs)
# to NL
# a df contain potential IMOs and its sequence travel from RU to NL port
pot_imo_from_RU_to_NL = pd.DataFrame(columns = snd_IMO_were_to_NL_df.columns)
for row in range(len(snd_IMO_were_to_NL_df)):
    imo_from_2ndport_to_NL = alltankers_adjusted[
        alltankers_adjusted['IMO'] == snd_IMO_were_to_NL_df['IMO'].iloc[row]]
    time_imo_at_NL_ports = imo_from_2ndport_to_NL[imo_from_2ndport_to_NL['DepPort'].isin(NL_ports)]
    time_imo_at_NL_ports = time_imo_at_NL_ports['ArrDate']
    if any(snd_IMO_were_to_NL_df['DepDate'].iloc[row] < time_imo_at_NL_ports):
        pot_imo_from_RU_to_NL = pd.concat([pot_imo_from_RU_to_NL,
                                           snd_IMO_were_to_NL_df.iloc[row: row+1]])
    else:
        next
# extract only routes that directly travek from 2nd port to NL or
# stop by more ports before arriving in NL but no more oil transhipment-the 
# same ship from 2nd port to the Netherlands
# A condition for a ship that stop at more than one ports, should not visit RU 
# again or visit a port included in the route more than twice

# select one hop
# select more hops but check conditions
# check in the dataframe from the timestamp at the 2nd ports to the time it
# reaches the Netherlands for the first time 

# filter route from 2nd port to NL, spatial consistency (not go back to RU or
# revisit a node twice) and temporal consistency ( time arrive in NL is later
# than time leave the second port)
one_oil_trans_fr_RU_to_NL = []
for row in range(len(pot_imo_from_RU_to_NL)):
    # extract only sequence of a potential IMO
    pot_imo = alltankers_adjusted[alltankers_adjusted['IMO'] ==
                                  pot_imo_from_RU_to_NL['IMO'].iloc[row]].reset_index(drop = True)
    # locate index of oil transshipment at the 2nd port based on deptime, 
    # depport, arridate
    snd_target_port = pot_imo[(pot_imo['DepPort'] == pot_imo_from_RU_to_NL['DepPort'].iloc[row])
                                        & (pot_imo['DepDate'] == pot_imo_from_RU_to_NL['DepDate'].iloc[row])
                                        & (pot_imo['ArrDate'] == pot_imo_from_RU_to_NL['ArrDate'].iloc[row])]
    # if it is a direct route to NL save, else only extract trip segment from
    # the moment of potential oil transshipment to the first time it arrives
    # in the NL
    one_oil_trans = pd.DataFrame(columns = alltankers_adjusted.columns)
    if snd_target_port['ArrPort'].isin(NL_ports).any():
        one_oil_trans = pd.concat([one_oil_trans, snd_target_port])
    else:
        row_nr_of_target_2nd_port = snd_target_port.index[0]
        
        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:len(pot_imo)]
        first_nl_port = pot_imo_from_2nd_target_port[
            pot_imo_from_2nd_target_port['DepPort'].isin(NL_ports)].index[0]
        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:first_nl_port]
        if ~pot_imo_from_2nd_target_port['DepPort'].isin(port_of_russia).any():
            # create network
            edges = []
            for n in range(len(pot_imo_from_2nd_target_port)):
                info = tuple([pot_imo_from_2nd_target_port['DepPort'].iloc[n], 
                              pot_imo_from_2nd_target_port['ArrPort'].iloc[n]])
                edges.append(info)
            # create graph
            ## multi-direct-graph
            Graph_seg_route = nx.MultiDiGraph()
            Graph_seg_route.add_edges_from(edges)
            # remove loop
            Graph_seg_route.remove_edges_from(list(nx.selfloop_edges(Graph_seg_route)))
            nodes_degree = list(Graph_seg_route.degree())
            # check degree for each nodes in the trip segment
            node_with_more_2_degree = [node for node, value in nodes_degree if value >=3]
            # if there are nodes with > 2 degree, the route is invalid. Otherwise, append the route
            if len(node_with_more_2_degree) == 0:
                one_oil_trans = pd.concat([one_oil_trans,
                                                       pot_imo_from_2nd_target_port])
    if len(one_oil_trans) != 0:
        one_oil_trans_fr_RU_to_NL.append(one_oil_trans)
                
temp_one_oil_trans_fr_RU_to_NL = pd.concat(one_oil_trans_fr_RU_to_NL, ignore_index=True)
# retrieve route from RU to the 2nd port and make a connection with the selected IMO
# available at the second port
route_fr_RU_2ndPort_fn = []

imo_unique_fr_RU_to_NL = temp_one_oil_trans_fr_RU_to_NL['IMO'].unique()


for i in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
    seg_df = track_route_fr_RU_to_2ndPort_and_connected_IMO[i]
    if seg_df['IMO'].iloc[1] in imo_unique_fr_RU_to_NL:
        route_fr_RU_2ndPort_fn.append(seg_df)


# Final combined list
combined_routes = []

for df2 in route_fr_RU_2ndPort_fn:
    for df1 in one_oil_trans_fr_RU_to_NL:
        if df2.iloc[-1].equals(df1.iloc[0]):
            # Drop first row of df1 to avoid duplicate
            merged = pd.concat([df2, df1.iloc[1:]], ignore_index=True)
            combined_routes.append(merged)

