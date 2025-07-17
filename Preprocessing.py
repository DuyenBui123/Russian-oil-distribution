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
                                         scnd_in_day, low_t_time, up_t_time,start_IMO,cons_IMO,cons_IMO_nr)

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
            
            
    start_IMO_v2 = {} # start from a specific RU port. Expect a dict of dict. with the
    # first layer contain RU port names and its connected IMO info in general.
    # The secondlayer key: nb name and its attributes
    cons_IMO_v2 = {} # the next second port met conditions (no EU port) and contain a connected IMO
    cons_IMO_nr_v2 = [] # contain a connected IMO        
      
    tracking_all_nbs_next_oiltrans_connect_IMO = []
           

    for row_in_imo_list in range(len(tracking_filtered_snd_oiltransshipment_imo_list)): #tracking_snd_oiltransshipment_imo_list -has more row and each row is a different label
        

        # loop through all neighbours of Novorossiysk
        

        # extract route from a RU port to its neighbout
        route_from_RUport_to_its_nb = tracking_filtered_snd_oiltransshipment_imo_list[row_in_imo_list].iloc[1:2]
        arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
        # extract all IMO available at the arrival port of the first trip from RU
        diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(arr_port)]
        # for each nb, calculate time gap between IMO from RU to 2nd port and 
        # the IMO available at the 2nd port
        tracking_all_nbs_next_oiltrans_connect_IMO, start_IMO_v1, cons_IMO_v1, cons_IMO_nr_v1 = pr.potential_IMO_at_cons_shared_port(
            tracking_filtered_snd_oiltransshipment_imo_list, 
                                                  row_in_imo_list, 
                                                  tracking_all_nbs_next_oiltrans_connect_IMO, 
                                             route_from_RUport_to_its_nb, start_RU_port, 
                                             diff_IMO_at_2ndPort, 
                                             scnd_in_day, low_t_time, up_t_time,start_IMO_v2,cons_IMO_v2,cons_IMO_nr_v2)
    # update
    start_IMO = start_IMO_v1
    cons_IMO = cons_IMO_v1
    cons_IMO_nr = cons_IMO_nr_v1
    track_route_fr_RU_to_2ndPort_and_connected_IMO = []
    track_route_fr_RU_to_2ndPort_and_connected_IMO = tracking_all_nbs_next_oiltrans_connect_IMO
    n = n+1



    
# Phase 2   
# unique set of connected IMO considering from all nbs
cons_IMO_nr_uniq = set(cons_IMO_nr)

last_IMO_in_seq = []
for lst in track_route_fr_RU_to_2ndPort_and_connected_IMO:
    last_IMO_in_seq.append(lst['IMO'].iloc[-1])
    
last_IMO_in_seq = set(last_IMO_in_seq)

# identify which connected IMO were to the NL
cons_IMO_were_to_NL = last_IMO_in_seq.intersection(IMO_in_NL['IMO'].unique())


# loop through a dict of connected IMO at nb nodes of the first RU trip. Only 
# select connected IMO that were to NL
snd_IMO_were_to_NL_df = pd.DataFrame()

for key, value in cons_IMO.items():


            for ind, timestamp in value:
                IMO_snd_port = timestamp['IMO']
                print(IMO_snd_port)
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
        # potential IMO from RU to NL with sequence of the snd port to the first NL port
        pot_imo_from_2nd_target_port = pot_imo[row_nr_of_target_2nd_port:first_nl_port]
        # if there is no RU port in that sequence then create a network
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
    if seg_df['IMO'].iloc[-1] in imo_unique_fr_RU_to_NL:
        route_fr_RU_2ndPort_fn.append(seg_df)


# Final combined list
combined_routes = []

for df2 in route_fr_RU_2ndPort_fn:
    for df1 in one_oil_trans_fr_RU_to_NL:
        if df2.iloc[-1].equals(df1.iloc[0]):
            # Drop first row of df1 to avoid duplicate
            merged = pd.concat([df2, df1.iloc[1:]], ignore_index=True)
            combined_routes.append(merged)

combined_routes = []
for i in range(len(track_route_fr_RU_to_2ndPort_and_connected_IMO)):
    seg_df = track_route_fr_RU_to_2ndPort_and_connected_IMO[i]
    for j in range(len(one_oil_trans_fr_RU_to_NL)):
        seg_filtered = one_oil_trans_fr_RU_to_NL[j]
        if (seg_df['IMO'].iloc[-1] == seg_filtered['IMO'].iloc[-1]) :
            complete_route = pd.concat([track_route_fr_RU_to_2ndPort_and_connected_IMO[i],one_oil_trans_fr_RU_to_NL[j]])
            combined_routes.append(complete_route)

combined_routes = [lst.drop_duplicates() for lst in combined_routes] 



# add round 2 - the second oil transshipment

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
        
        
start_IMO_v1 = {} # start from a specific RU port. Expect a dict of dict. with the
# first layer contain RU port names and its connected IMO info in general.
# The secondlayer key: nb name and its attributes
cons_IMO_v1 = {} # the next second port met conditions (no EU port) and contain a connected IMO
cons_IMO_nr_v1 = [] # contain a connected IMO        
tracking_each_nb_next_oiltrans_connect_IMO = []        
tracking_all_nbs_next_oiltrans_connect_IMO = []
for row_in_imo_list in range(len(tracking_filtered_snd_oiltransshipment_imo_list)): #tracking_snd_oiltransshipment_imo_list -has more row and each row is a different label
    

    # loop through all neighbours of Novorossiysk
    

    # extract route from a RU port to its neighbout
    route_from_RUport_to_its_nb = tracking_filtered_snd_oiltransshipment_imo_list[row_in_imo_list].iloc[1:2]
    arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
    # extract all IMO available at the arrival port of the first trip from RU
    diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(arr_port)]
    # for each nb, calculate time gap between IMO from RU to 2nd port and 
    # the IMO available at the 2nd port
    
    for row in range(len(route_from_RUport_to_its_nb)):
        
        # arrive time of IMO travel from RU to its nb
        arr_time_of_IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[row,
                                              route_from_RUport_to_its_nb.columns.get_loc('ArrDate')]
        arr_port_per_row = route_from_RUport_to_its_nb.iloc[row,
                                              route_from_RUport_to_its_nb.columns.get_loc('ArrPort')]
        IMO_fr_RU_to_2ndPort = route_from_RUport_to_its_nb.iloc[0:2]
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


                test_test_ = pd.concat([tracking_filtered_snd_oiltransshipment_imo_list[row_in_imo_list],
                                        IMO_avai_at_2ndPort])
                if len(test_test_) != 0:
                    
                    tracking_each_nb_next_oiltrans_connect_IMO.append(test_test_)
                index = 1
                # check whether the key(nb name) in the dict, not create new. 
                # Otherwise, append 
                if dep_port_per_row not in list(cons_IMO_v1.keys()):
                    cons_IMO_v1[dep_port_per_row] =  tuple([[
                        index, {'DepPort' : diff_IMO_at_2ndPort['DepPort'].iloc[row_dep_port],
                                'ArrPort' : diff_IMO_at_2ndPort['ArrPort'].iloc[row_dep_port],
                                'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                      'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                      'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                    IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                    cons_IMO_nr_v1.append(IMO)
                else:
                    index = len(cons_IMO_v1[dep_port_per_row]) + 1  
                    cons_IMO_v1[dep_port_per_row] += tuple([[
                        index, {'DepPort' : diff_IMO_at_2ndPort['DepPort'].iloc[row_dep_port],
                                'ArrPort' : diff_IMO_at_2ndPort['ArrPort'].iloc[row_dep_port],
                                'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[row_dep_port],
                      'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[row_dep_port],
                      'IMO': diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]}]])
                    IMO = diff_IMO_at_2ndPort['IMO'].iloc[row_dep_port]
                    cons_IMO_nr_v1.append(IMO)
            else:
                next


    # add main key of the dictionary, the key is a RU port name
    start_IMO_v1['Novorrosiyk'] = cons_IMO_v1

         
        
      
        
        
   
    




            


# check the pattern of each trip (ploting? - how to see the pattern? find recurring movement?-count how many time?)
# upgrade this into 2 oil transhipments  

# check again the IMO selected from 2nd to NL because Zhelezny Rog Port does not lead to NL directly

# cons_IMO['Aliaga']
a = alltankers_adjusted[alltankers_adjusted['IMO'] == 9933573]
# inter_9749506 =  set(a['DepPort']).intersection(neighbors_Nov)
#     len(cons_IMO[edges_Nov[1][1]]/2)               
#    tuple([{'IMO': diff_IMO_at_2ndPort.iloc[0, 0], 'DepPort' :diff_IMO_at_2ndPort.iloc[0, 1]}])         
# tuple([0, {'DepDate' : diff_IMO_at_2ndPort['DepDate'].iloc[0],
#  'ArrDate' : diff_IMO_at_2ndPort['ArrDate'].iloc[0],
#  'IMO': diff_IMO_at_2ndPort['IMO'].iloc[0]}])

# b = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(['Novorossiysk']) & 
#                     alltankers_adjusted['ArrPort'].isin(['Sikka'])]
# calculate time different and identify IMO in the first ports
# remove EU port in the second visited port sequence
# loop through each neighbour, colect IMO, 
# extract 2 hops








# # TEST get paths from two given nodes
# Graph_whole_dataset.get_edge_data('Novorossiysk', 'Aliaga' )
# list(Graph_whole_dataset.neighbors('Aliaga' ))
# direct_graph.get_edge_data('Vlissingen', 'Antwerp')
# nx.has_path(direct_graph, 'Novorossiysk', 'Vlissingen')
# paths_nov_vlis = list(nx.all_simple_paths(direct_graph, source='Novorossiysk', target='Rotterdam', cutoff = 10)) #cutoff
# paths_nov_vlis = list(nx.all_simple_paths(Graph_whole_dataset, source='Novorossiysk', target='Vlissingen')) #cutoff
# paths_nov_vlis = list(nx.all_shortest_paths(Graph_whole_dataset, source='Novorossiysk', target='Vlissingen'))

# ## get all paths from RU to NL
# all_paths_RU_NL= []
# for comb_ru_nl_path in comb_ru_nl_ports:
#     print(comb_ru_nl_path, comb_ru_nl_path[0], comb_ru_nl_path[1])
#     paths = list(nx.all_simple_paths(Graph_whole_dataset, source=comb_ru_nl_path[0], target=comb_ru_nl_path[1]))
#     print(paths)
#     all_paths_RU_NL = all_paths_RU_NL.append(paths)
# # try with parallel computing
# def find_route_RU_NL(x):
#     print(x, x[0], x[1])
#     paths = list(nx.all_simple_paths(Graph_whole_dataset, source=x[0], target=[1]))

#     return paths

# if __name__ == '__main__':
#     with Pool(5) as p:
#         print(p.map(find_route_RU_NL, comb_ru_nl_ports))
        
# # try parallel computing with neighbours of a given port
     
# def find_paths_from_neighbor(args):
#     G, neighbor, target, cutoff = args
#     try:
#         return list(nx.all_simple_paths(G, source=neighbor, target=target, cutoff=cutoff))
#     except Exception:
#         return []

# def parallel_all_simple_paths(G, source, target, cutoff=10, max_paths=None):
#     neighbors = list(G.neighbors(source))
#     filter_neighbors = [n for n in neighbors if n not in eu_ports]
#     agg = [(G, neighbor, target, cutoff) for neighbor in filter_neighbors]
#     partial_paths = []

#     with Pool(processes=cpu_count()) as pool:
#         results = pool.map(find_paths_from_neighbor, agg)

#     for paths in results:
#         for path in paths:
#             full_path = [source] + path
#             partial_paths.append(full_path)
#             if max_paths and len(partial_paths) >= max_paths:
#                 return partial_paths

#     return partial_paths

# # Run it
# paths = parallel_all_simple_paths(Graph_whole_dataset, 'Novorossiysk', 'Vlissingen', cutoff=5, max_paths=10)
# Nov_neighbour= list(Graph_whole_dataset.neighbors('Novorossiysk'))
# Nov_neighbour_filter = [n for n in Nov_neighbour if n not in eu_ports]
# # OLD CODE
# # # read portname file with unloco and coordinate
# # portname_unloco_coor = pd.read_csv('./preprocessing/inter_input/portname_unloco.txt', delimiter=',')
# # %% start of assigning lat long

# # # standadize port name
# alltankers_adjusted['DepPort'] = alltankers_adjusted['DepPort'].map(lambda x: 
#                                                                     pp.standardize_port_name(x))
# alltankers_adjusted['ArrPort'] = alltankers_adjusted['ArrPort'].map(lambda x: 
#                                                                     pp.standardize_port_name(x))

# def move_fso_fpso_to_end(name):
#     name = str(name).strip()
#     # Search for FSO or FPSO
#     match = re.search(r'\b(FSO|FPSO)\b', name, flags=re.IGNORECASE)
#     if match:
#         # Remove FSO/FPSO and re-add at the end
#         cleaned = re.sub(r'\b(FSO|FPSO)\b', '', name, flags=re.IGNORECASE).strip()
#         return f"{cleaned} {match.group(1).upper()}".strip()
#     return name

# # Example usage:
# alltankers_adjusted['DepPort'] = alltankers_adjusted['DepPort'].apply(move_fso_fpso_to_end)  
# # alltankers_adjusted['DepPort'].unique()  
# # portname, countport = np.unique(tanker_were_to_NL_ww['PORTNAME'], return_counts = True)
# # len(portname)

# # harbor_names = pd.read_csv("./Data/IVS3_CodeTabel_Plaatsen.csv")
# # extra_harbor_names = pd.read_csv('./Data/port_unloco.csv')
# # port_names = pd.concat([harbor_names, extra_harbor_names], ignore_index=True, sort=False)

# # columns_to_keep = ['Code', 'Plaats']
# # harbor_names = port_names[columns_to_keep]
# # port_names.loc[:,'Plaats'] = port_names['Plaats'].str.replace(r'\s*\(.*?\)', '', regex=True)

# # alltankers_adjusted['DepPort'] =alltankers_adjusted['DepPort'].str.upper()
# # alltankers_adjusted['ArrPort'] =alltankers_adjusted['ArrPort'].str.upper()
# #     # Merge (geo)dataframes
# # alltankers_adjusted_ww_with_codes = alltankers_adjusted.merge(port_names,
# #         left_on='DepPort',
# #         right_on='Plaats',
# #         how='left')
# # mask = (alltankers_adjusted_ww_with_codes['Code'] == 'USRAJ') & \
# #        (alltankers_adjusted_ww_with_codes['Country'] == 'Netherlands')

# # alltankers_adjusted_ww_with_codes.loc[mask, 'Code'] = 'NLRTM'

# # Official dataset (downloaded from DataHub)
# ref = pd.read_csv('./Data/code-list.csv')  # Coordinates already in decimal
# # Load reference UN/LOCODE list

# ref['UNLOCO'] = ref['Country'] + ref['Location']
# ref[['Latitude_ref','Longitude_ref']] = ref['Coordinates'].apply(
#     lambda c: pd.Series(pp.parse_unloco_coords(c))
# )
# country_codes = pd.read_csv('./Data/country-codes.csv')  # This should contain country code-to-name mapping

# # Inspect the country code column names (update if different)
# # For example, assuming:
# # - ports_df has a column 'LandCode'
# # - country_codes has columns 'Code' and 'CountryName'

# # Rename for clarity if needed
# country_codes = country_codes.rename(columns={'CountryCode': 'Country'})
# country_codes.loc[:,'CountryName'] = country_codes['CountryName'].str.replace(r'\s*\(.*?\)', '', regex=True)
# mask = (country_codes['CountryName'] == 'Russian Federation')

# country_codes.loc[mask, 'CountryName'] = 'Russia'
# # Remove accents
# def remove_accents(text):
#     if isinstance(text, str):
#         return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
#     return text
# # def clean_name(name):
# #     if pd.isna(name):
# #         return name
# #     # Remove diacritics (accents)
# #     name = ''.join(
# #         c for c in unicodedata.normalize('NFD', name)
# #         if unicodedata.category(c) != 'Mn'
# #     )
# #     # Replace or remove special characters
# #     name = name.replace("'", "").replace("/", " ").replace('"', "").name.replace("-", " ").
# #     name = name.replace("-", " ")
# #     # Replace multiple spaces with a single space
# #     name = re.sub(r'\s+', ' ', name)
# #     return name.strip()
# country_codes['CountryName'] = country_codes['CountryName'].apply(clean_name)
# a = sorted(country_codes['CountryName'].unique())
# # Fix known country name mappings
# def normalize_country(name):
#     name = remove_accents(name)
#     country_map = {
#         'Turkiye': 'Turkey',
#         'Cote dIvoire': "Cote d'Ivoire",
#         'Aland Islands': 'Aland Islands',
#         'Saint Barthelemy': 'Saint Barthelemy'
#         "Vassilevsky OstrovSt Petersburg" : 'St Petersburg',
#         "Ust-Luga" : 'Ust Luga'
#         # Add more if needed
#     }
#     return country_map.get(name, name)
# def normalize_NameWoDiacritics(name):

#     country_map = {

#         "Vassilevsky Ostrov\St Petersburg" : 'St Petersburg',
#         "Ust'-Luga" : 'Ust Luga'
#         # Add more if needed
#     }
    
    
#     return country_map.get(name, name)

# country_codes['CountryName'] = country_codes['CountryName'].apply(normalize_country)
# # Merge to add full country name
# merged_ref = ref.merge(country_codes, on='Country', how='left')
# merged_ref['NameWoDiacritics'] = merged_ref['NameWoDiacritics'].str.upper()
# merged_ref['NameWoDiacritics'] = merged_ref['NameWoDiacritics'].apply(normalize_NameWoDiacritics)
# #merged_ref['NameWoDiacritics'] = merged_ref['NameWoDiacritics'].astype(str).str.split(',').str[0]

# merged_ref = merged_ref[~merged_ref.duplicated(subset=['NameWoDiacritics', 'CountryName'], keep=True)]
# alltankers_adjusted['DepPort'] =alltankers_adjusted['DepPort'].str.upper()
# alltankers_adjusted['ArrPort'] =alltankers_adjusted['ArrPort'].str.upper()
# alltankers_adjusted_ww_with_codes = alltankers_adjusted.merge(merged_ref,
#         left_on=['DepPort', 'Country'],
#         right_on=['NameWoDiacritics','CountryName'],
#         how='left')
# # Remove columns
# alltankers_adjusted_ww_with_codes = alltankers_adjusted_ww_with_codes[['IMO',
#                                             'DepPort', 'ArrPort', 'DepDate', 'ArrDate',
#                                             'ShipType', 'Country_x', 'TravelTime',
#                                             'BerthTime', 'UNLOCO', 'Latitude_ref',
#                                             'Longitude_ref']]

# # Merge missing data
# port_names = pd.read_csv("./Data/IVS3_CodeTabel_Plaatsen.csv")
# #extra_harbor_names = pd.read_csv('./Data/port_unloco.csv')
# #port_names = pd.concat([harbor_names, extra_harbor_names], ignore_index=True, sort=False)

# columns_to_keep = ['Code', 'Plaats']
# harbor_names = port_names[columns_to_keep]
# port_names.loc[:,'Plaats'] = port_names['Plaats'].str.replace(r'\s*\(.*?\)', '', regex=True)

# # Only rows where UNLOCO is missing
# alltankers_adjusted_ww_with_codes_missing = alltankers_adjusted_ww_with_codes[alltankers_adjusted_ww_with_codes['UNLOCO'].isna()]

# # Merge missing UNLOCO rows with lookup table
# alltankers_adjusted_ww_with_codes_missing_merged = alltankers_adjusted_ww_with_codes_missing.merge(port_names,
#     left_on='DepPort', right_on='Plaats', how='left', suffixes=('', '_new'))

# # Fill missing UNLOCO values with the new matched values
# alltankers_adjusted_ww_with_codes.loc[alltankers_adjusted_ww_with_codes['UNLOCO'].isna(),
#                                       'UNLOCO'] = alltankers_adjusted_ww_with_codes_missing_merged['Code']


# # Step 1: Create a mapping from DepPort to known UNLOCO
# port_to_unloco = alltankers_adjusted_ww_with_codes[alltankers_adjusted_ww_with_codes['UNLOCO'].notna()].groupby('DepPort')['UNLOCO'].first()

# # Step 2: Fill UNLOCO if it's missing AND there is a known value for the same DepPort
# alltankers_adjusted_ww_with_codes['UNLOCO'] = alltankers_adjusted_ww_with_codes.apply(
#     lambda row: port_to_unloco[row['DepPort']] if pd.isna(row['UNLOCO']) and row['DepPort'] in port_to_unloco else row['UNLOCO'],
#     axis=1
# )
# a = ref[ref['NameWoDiacritics'] == 'Houston']
# a = ref[ref['NameWoDiacritics'] == 'Novorossiysk']
# a = merged_ref[merged_ref['NameWoDiacritics'] == 'Novorossiysk']
# b = alltankers_adjusted[alltankers_adjusted['DepPort'] == 'HOUSTON']
# # Merge on UN/LOCODE code
# alltankers_adjusted_ww_with_codes = alltankers_adjusted_ww_with_codes.merge(ref[['UNLOCO','Latitude_ref','Longitude_ref']],
#               left_on='Code', right_on='UNLOCO', how='left')

# # Fill missing with 0.00
# alltankers_adjusted_ww_with_codes['Latitude']  = alltankers_adjusted_ww_with_codes['Latitude_ref'].fillna(0.00)
# alltankers_adjusted_ww_with_codes['Longitude'] = alltankers_adjusted_ww_with_codes['Longitude_ref'].fillna(0.00)

# # Drop helper columns
# df = df.drop(columns=['UNLOCO','Latitude_ref','Longitude_ref','Coordinates'])
# # %% end of assigning lat and long
# # alltankers.loc[:, ['COUNTRY ', 'PORTNAME']].drop_duplicates().to_csv('./preprocessing/pp_inter_ouput/portname.csv')
# # Prepare edge and edge attributes
# # For the whole graph

# edges_and_attributes = []
# for n in range(len(alltankers_adjusted)):
#     info = tuple([alltankers_adjusted['DepPort'].iloc[n], 
#                   alltankers_adjusted['ArrPort'].iloc[n],
#                   {'DepDate' : str(alltankers_adjusted['DepDate'].iloc[n]),
#                    'ArrDate' : str(alltankers_adjusted['ArrDate'].iloc[n]),
#                    'TravelTime' : str(alltankers_adjusted['TravelTime'].iloc[n]),
#                    'IMO': alltankers_adjusted['IMO'].iloc[n]}])
#     edges_and_attributes.append(info)
# Graph_whole = nx.MultiDiGraph()
# Graph_whole.add_edges_from(edges_and_attributes)


# # Get in-degree (number of incoming edges) for each node
# in_degrees = dict(Graph_whole.in_degree())

# sorted_in_degrees = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:-10]
# low_node_in_degree = {}
# for node, values in in_degrees.items():
#     if values <=1:
#         low_node_in_degree[node] = values
# # Get out-degree (number of outcoming edges) for each node
# out_degrees = dict(Graph_whole.out_degree())
# sorted_out_degrees = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:-10]
# low_node_outdegree = {}
# for node, values in out_degrees.items():
#     if values <=1:
#         low_node_outdegree[node] = values
# degree = {}
# for node1, value in in_degrees.items():
#     degree[node1] = value + out_degrees[node1]
# # get out node only used or transeve through 1 times
# low_node_degree = {}
# for node, value in degree.items():
#     if value <=2:
#         low_node_degree[node] = value
        

# # Find most and least popular nodes
# most_popular = max(in_degrees, key=in_degrees.get)
# least_popular = min(in_degrees, key=in_degrees.get)
# # count edges
# edges_counts = Counter((u, v) for u, v in Graph_whole.edges())

# counts, bins = np.histogram(np.array(list(edges_counts.values()), dtype=float),
#                             bins = 100, range = (0,40))
# plt.stairs(counts, bins)
# plt.xlabel("Nr of edges frequency")
# plt.ylabel("Count")
# plt.title("The frequency of each edge")
# plt.show
# # frequency of edge
# low_frequency_edges=[]
# for edge, value in edges_counts.items():
#     if value <= 1:
#         low_frequency_edges.append(edge)
# # determin nodes that has selfloop
# # Count nodes has selfloop
# self_loops = list(nx.selfloop_edges(Graph_whole, keys=True))
# nodes_to_check = set(u for u, v, k in self_loops)
# len(nodes_to_check)
# # For segments

# last_date = alltankers_adjusted['DepDate'].iloc[-1].normalize()
# n = 1
# while n >0 :
#     if n == 1:
#         start_date = str(alltankers_adjusted['DepDate'].iloc[1].date())
#         next_iter_day = pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d") +
#                                      timedelta(days = 5)).normalize()
#         start_date = alltankers_adjusted['DepDate'].iloc[1].normalize()
#         seg_tankers = alltankers_adjusted[(
#             alltankers_adjusted['DepDate'] >= start_date)
#             & (alltankers_adjusted['DepDate'] <= next_iter_day)]
                                      
#         edges_and_attributes = []
#         for n in range(len(seg_tankers)):
#             info = tuple([seg_tankers['DepPort'].iloc[n], 
#                           seg_tankers['ArrPort'].iloc[n],
#                           {'DepDate' : str(seg_tankers['DepDate'].iloc[n]),
#                            'ArrDate' : str(seg_tankers['ArrDate'].iloc[n]),
#                            'TravelTime' : str(seg_tankers['TravelTime'].iloc[n]),
#                            'IMO': seg_tankers['IMO'].iloc[n]}])
#             edges_and_attributes.append(info)
            
        
#         # plot graph
        
#         # Work on smaller skill
#         # kamada_kawai_layout
        
#         Graph = nx.MultiDiGraph()
#         Graph.add_edges_from(edges_and_attributes)
#         # pos = nx.kamada_kawai_layout(Graph)
#         pathes = list(nx.all_simple_paths(Graph, source='Novorossiysk', target='Vlissingen', cutoff= 12))
#         pathes[1]
#         # check the temporal consistancy of the collected paths
#         for n in range(len(pathes[1])-1):
#             path_info = []

#             edge_data = Graph.get_edge_data(pathes[1][n], pathes[1][n+1])
#             path_info.append((pathes[1][n], pathes[1][n+1], edge_data))
#             print("Edge attributes in this path:")
#             for edge in path_info:
#                 print(edge)
#         list(Graph.neighbors('Novorossiysk'))
#         a = list(nx.dfs_edges(Graph, source='Novorossiysk', depth_limit=10))
        
# from multiprocessing import Pool, cpu_count
# from itertools import islice        
# def find_paths_from_neighbor(neighbor):
#     """Wrapper to find all simple paths from source through a specific neighbor."""
#     try:
#         return list(nx.all_simple_paths(Graph, neighbor, target, cutoff=cutoff - 1))
#     except:
#         return []

# def parallel_all_simple_paths(source, target, cutoff=10, max_paths=None):
#     neighbors = list(Graph.neighbors(source))
#     partial_paths = []

#     with Pool(processes=cpu_count()) as pool:
#         results = pool.map(find_paths_from_neighbor, neighbors)

#     # Prepend the source node to each path and flatten
#     for paths in results:
#         for path in paths:
#             full_path = [source] + path
#             partial_paths.append(full_path)
#             if max_paths and len(partial_paths) >= max_paths:
#                 return partial_paths

#     return partial_paths

# # Run it
# paths = parallel_all_simple_paths('Novorossiysk', 'Vlissingen', cutoff=5, max_paths=10)

# print(f"Found {len(paths)} paths")
#         # check the temporal consistancy of the collected paths
#         for n in range(len(longest_path)-1):
#             path_info = []

#             edge_data = Graph.get_edge_data(longest_path[n], longest_path[n+1])
#             path_info.append((longest_path[n], longest_path[n+1], edge_data))
#             print("Edge attributes in this path:")
#             for edge in path_info:
#                 print(edge)
#         # analyse the graph
#         # define nodes of the trip segments that in the port of interest

#         graph_nodes = list(Graph.nodes())
#         match_ru_node = []
#         for node_ru in port_of_russia:
#             if node_ru in graph_nodes:
#                 match_ru_node.append(node_ru)
#         match_3rd_refinery_nodes = []        
#         for node_3rd in ports_of_interest:
#             if node_3rd in graph_nodes:
#                 match_3rd_refinery_nodes.append(node_3rd)
#         # check for available edges between two nodes
        
#         # Define all path from one port of interst to another
#         # An example of two know source and target
# def find_temporal_paths(G, source, target, max_depth=10):
#     all_paths = []

#     def dfs(path, current_node, current_time):
#         if len(path) > max_depth:
#             return
#         if current_node == target:
#             all_paths.append(list(path))
#             return
#         for neighbor in G.successors(current_node):
#             for key, data in G.get_edge_data(current_node, neighbor).items():
#                 dep_time = data['DepDate']
#                 arr_time = data['ArrDate']
#                 if dep_time >= current_time:
#                     path.append((neighbor, dep_time, arr_time))
#                     dfs(path, neighbor, arr_time)
#                     path.pop()

#     for _, neighbor, data in G.out_edges(source, data=True):
#         dep_time = data['DepDate']
#         arr_time = data['ArrDate']
#         dfs([(source, dep_time, arr_time), (neighbor, dep_time, arr_time)], neighbor, arr_time)

#     return all_paths
# a = find_temporal_paths(Graph_whole, 'Novorossiysk', 'Yarimca', max_depth =10)

#         Nov_yarim_pathes = list(nx.all_simple_paths(Graph, 
#                                                     source='Novorossiysk', 
#                                                     target='Yarimca'))       
#         for path in Nov_yarim_pathes:
#             for node in range(len(path) -1):
#                     u, v = path[node], path[node + 1]
#                     u_, v_ = path[node+2], path[node + 3]
#                 for k in Graph[u][v]:
#                 for k_ in Graph[u_][v_]:
#                     depdate = Graph[u][v][k].get('DepDate')
#                     depdate_ = Graph[u_][v_][k_].get('DepDate')
#                     compare depdate and depdate_
#         # define longest path
#         longest_path = list(list(max(nx.all_simple_paths(Graph, 
#                                                          source='Novorossiysk',
#                                                          target='Yarimca'), key=lambda x: len(x))))
#         # check the temporal consistancy of the collected paths
#         for n in range(len(longest_path)-1):
#             path_info = []

#             edge_data = Graph.get_edge_data(longest_path[n], longest_path[n+1])
#             path_info.append((longest_path[n], longest_path[n+1], edge_data))
#             print("Edge attributes in this path:")
#             for edge in path_info:
#                 print(edge)
                
                
#         valid_paths = list(set(tuple(path) for path in valid_paths))
#             path_info = []
#             for n in range(len(list(valid_paths[1]))-1):
#                 path_info = []

    
#                 edge_data = Graph[valid_paths[1][n]][valid_paths[1][n+1]]
#                 print(edge_data)
    
#                 path_info_tup = tuple([Graph[valid_paths[1][n]],Graph[valid_paths[1][n+1]], edge_data])
#                 path_info.append(path_info_tup)

#         # Define shortest paths
#         shortest_path = list(nx.all_shortest_paths(Graph, 
#                                                    source='Novorossiysk',
#                                                    target='Yarimca'))
#         # define graphs for each path
#         H = nx.MultiDiGraph()
#         S_p = nx.MultiDiGraph()
#         L_p = nx.MultiDiGraph()
#         #H.add_nodes_from(Graph.nodes)
#         # add edges for the new graphs

#         for cyc in Nov_yarim_pathes:
#             for a, b in zip(cyc, cyc[1:]):
#                 H.add_edge(a, b)
#         for n in range(len(longest_path)-1):

#                 L_p.add_edge(longest_path[n], longest_path[n+1])
#                 L_p.edges()
#         for cyc in shortest_path:
#             for a, b in zip(cyc, cyc[1:]):
#                 S_p.add_edge(a, b)
#         # def get_long_color_list(n, cmap_name='nipy_spectral'):
#         #     cmap = plt.get_cmap(cmap_name, n)
#         #     return [mcolors.to_hex(cmap(i)) for i in range(n)]
        
#         # color_list = get_long_color_list(100, 'nipy_spectral')  # up to 256+ safe
#         # color_list = color_list[:len(Nov_yarim_pathes)]
#         # for path, color in zip(Nov_yarim_pathes, color_list):
#         #     print(f'this is one path {path}')
#         #     for edge in zip(path[:-1], path[1:]):
#         #         print(f' this is one segment {edge}')
#         #         H.add_edge(*edge, color=color)
#         # plot
#         plt.figure(figsize = (15, 10))
#         plt.subplot(1,3,1)
#         plt.title('all pathes')
#         pos = nx.kamada_kawai_layout(Graph)
#         # define node colors for specific node types
#         node_clr = ['pink' if node in ports_of_interest else 'black' if node in port_of_russia
#                     else 'blue' for node in H.nodes()]
#         edge_colors = nx.get_edge_attributes(H, 'color')
#         nx.draw(H, node_color = node_clr, pos=pos, 
#         with_labels=True, edgelist = H.edges(), arrowsize = 8)

#         plt.subplot(1,3,2)
#         plt.title('shortest pathes')
#         node_clr = ['pink' if node in ports_of_interest else 'black' if node in port_of_russia
#                     else 'blue' for node in S_p.nodes()]        

#         nx.draw(S_p, node_color = node_clr, pos=pos, 
#                 with_labels=True, edgelist = S_p.edges(), arrowsize = 8)
#         plt.subplot(1,3,3)
#         plt.title('longest pathes')
#         node_clr = ['pink' if node in ports_of_interest else 'black' if node in port_of_russia
#                     else 'blue' for node in L_p.nodes()]
        
#         pos = nx.kamada_kawai_layout(L_p)
#         nx.draw(L_p, node_color = node_clr, pos=pos, 
#                 with_labels=True, edgelist = L_p.edges(), arrowsize = 8)#edge_color = edge_colors.values()
#         plt.show()
            
        
#         # get edge info between two node
#         for key, edge_data in Graph.get_edge_data('Novorossiysk', 'Aliaga').items():
#             print(f"Edge key: {key}, data: {edge_data}")
#         graph_node_ru = graph_nodes in port_of_russia
#         alone_node = {}
#         node_names = list(Graph.nodes())
#         # remove nodes that stand alone or only have one eges
#         for node in node_names:
#             innode = Graph.in_edges(node)
#             outnode = Graph.out_edges(node)
#             totalnode = list(innode) + list(outnode)
#             if len(totalnode) <2:
#                 alone_node[node] = len(totalnode)
#         for node, value in alone_node.items():
#             Graph.remove_node(node)
        
#         print(f'NR node af remove {Graph.number_of_nodes()}')
#         print(f'NR edge af remove {Graph.number_of_edges()}')
#         # define and remove nodes with only selfloop
#         self_loops = list(nx.selfloop_edges(Graph, keys=True))
#         nodes_to_check = set(u for u, v, k in self_loops)
        
#         for node in nodes_to_check:
#             neighbors = set(n for u, n in Graph.edges(node) if n != node)
            
#             if not neighbors:
#                 # Iterate over a copy to avoid RuntimeError
#                 for u, v, k in list(Graph.edges(node, keys=True)):
#                     if u == v:  # self-loop
#                         Graph.remove_edge(u, v, key=k)
                            
                        
#         print(f'NR node af remove {Graph.number_of_nodes()}')
#         print(f'NR edge af remove {Graph.number_of_edges()}')
    
#         imo_gr = [k['IMO'] for n,m, k in Graph.edges(data=True)]
#         edge_freq = Counter(Graph.edges())
        
        
#         # Get unique IMOs
#         unique_imos = list(set(imo_gr))  # Ensure it's a list
#         num_colors = len(unique_imos)
        
#         # Generate long color list using a colormap (e.g., 'tab20', 'hsv', 'nipy_spectral')
#         cmap = plt.cm.get_cmap('tab20', num_colors)
#         color_list = [mcolors.to_hex(cmap(i)) for i in range(num_colors)]
        
#         color_map = {imo: color_list[i] for i, imo in enumerate(unique_imos)}
        
#         # Map edges to their colors
#         edge_colors = [color_map[data['IMO']] for u, v, data in Graph.edges(data=True)]
#         node_clr = ['pink' if node in ports_of_interest else 'black' if node in port_of_russia
#                     else 'blue' for node in Graph.nodes()]
#         pos = nx.spring_layout(Graph)
#         plt.figure(figsize = (12,8), dpi = 100)
#         nx.draw(Graph, pos, with_labels=True, edge_color=edge_colors, width=2,
#                 node_size = 80, node_color = node_clr)
#         # Create legend
        
#         legend_elements = [Patch(facecolor=color_map[imo], label=f'IMO {imo}') for imo in unique_imos]
#         plt.legend(handles=legend_elements, title="Edge IMO", loc='upper left', bbox_to_anchor=(1, 1))
        
#         plt.tight_layout()
#         plt.show()
#         n =+1
#     else:
#         new_start_date = next_iter_day - timedelta(days = 5)
#         next_iter_day = next_iter_day + timedelta(days = 5)
#         if (new_start_date < last_date) & (next_iter_day < last_date):
#             seg_tankers = alltankers_adjusted[(
#                 alltankers_adjusted['DepDate'] >= start_date)
#                 & (alltankers_adjusted['DepDate'] <= next_iter_day)]
                                          
#             edges_and_attributes = []
#             for n in range(len(seg_tankers)):
#                 info = tuple([seg_tankers['DepPort'].iloc[n], 
#                               seg_tankers['ArrPort'].iloc[n],
#                               {'DepDate' : str(seg_tankers['DepDate'].iloc[n]),
#                                'ArrDate' : str(seg_tankers['ArrDate'].iloc[n]),
#                                'TravelTime' : str(seg_tankers['TravelTime'].iloc[n]),
#                                'IMO': seg_tankers['IMO'].iloc[n]}])
#                 edges_and_attributes.append(info)
#             # plot graph
            
#             # Work on smaller skill
#             # kamada_kawai_layout
            
#             Graph = nx.MultiDiGraph()
#             Graph.add_edges_from(edges_and_attributes)
#             # pos = nx.kamada_kawai_layout(Graph)
#             # plt.figure(figsize = (16,12), dpi = 100)
#             # nx.draw_networkx(
#             #     Graph,
#             #     pos, arrowsize = 15, with_labels= False, node_size = 50, font_size = 8
#             # )
            
#             imo_gr = [k['IMO'] for n,m, k in Graph.edges(data=True)]
#             edge_freq = Counter(Graph.edges())

            
#             # Get unique IMOs
#             unique_imos = list(set(imo_gr))
            
#             # Assign a unique color to each
#             color_list = list(mcolors.TABLEAU_COLORS.values())
#             color_map = {imo: color_list[i % len(color_list)] for i, imo in enumerate(unique_imos)}
            
#             # Map edges to their colors
#             edge_colors = [color_map[data['IMO']] for u, v, data in Graph.edges(data=True)]
#             node_clr = ['pink' if node in ports_of_interest else 'black' if node in port_of_russia
#                         else 'blue' for node in Graph.nodes()]
#             pos = nx.kamada_kawai_layout(Graph)
#             plt.figure(figsize = (16,12), dpi = 100)
#             nx.draw(Graph, pos, with_labels=True, edge_color=edge_colors, width=2, node_color = node_clr)
#             # Create legend
            
#             legend_elements = [Patch(facecolor=color_map[imo], label=f'IMO {imo}') for imo in unique_imos]
#             plt.legend(handles=legend_elements, title="Edge IMO", loc='upper left', bbox_to_anchor=(1, 1))
            
#             plt.tight_layout()
#             plt.show()
        
#         else:
#             StopIteration
        
# # analyse the graph
# alone_node = {}
# node_names = list(Graph.nodes())
# for node in node_names:
#     innode = Graph.in_edges(node)
#     outnode = Graph.out_edges(node)
#     totalnode = list(innode) + list(outnode)
#     if len(totalnode) <2:
#         alone_node[node] = len(totalnode)
# for node, value in alone_node.items():
#     Graph.remove_node(node)

# Graph.number_of_nodes()
# Graph.number_of_edges()
# # Compare edges with different IMOs
# matches = []
# for nodename  in node_names:

#     mar_in = list(Graph.in_edges(nodename, data = True))
#     mar_out = list(Graph.out_edges(nodename, data = True))
#     edges = mar_in + mar_out
    
#     imo_unique = [val['IMO'] for n1, n2, val in edges]
#     imo_unique = list(set(imo_unique))
#     if len(imo_unique) >= 2:
#         # Parse edges
#         from datetime import datetime
#         from itertools import combinations
#         parsed_edges = []
#         for u, v, data in edges:
#             parsed_edges.append({
#                 'From': u,
#                 'To': v,
#                 'DepDate': datetime.strptime(data['DepDate'], "%Y-%m-%d %H:%M:%S"),
#                 'ArrDate': datetime.strptime(data['ArrDate'], "%Y-%m-%d %H:%M:%S"),
#                 'IMO': data['IMO']
#             })
#         # Threshold in seconds (e.g., 3 days)
#         threshold =  1*24* 60 * 60
        

        
#         for e1, e2 in combinations(parsed_edges, 2):
#             shared_port = (
#                 e1['From'] == e2['From'] or
#                 e1['From'] == e2['To'] or
#                 e1['To'] == e2['From'] or
#                 e1['To'] == e2['To']
#             )
#                         # Compare only if the ports involved in the time diff make sense
#             if e1['IMO'] != e2['IMO']:
#                 time_diffs = {
#                     'Dep–Dep': (
#                         abs((e1['DepDate'] - e2['DepDate']).total_seconds())
#                         if e1['From'] == e2['From'] else None
#                     ),
#                     'Arr–Arr': (
#                         abs((e1['ArrDate'] - e2['ArrDate']).total_seconds())
#                         if e1['To'] == e2['To'] else None
#                     ),
#                     'Arr–Dep': (
#                         abs((e1['ArrDate'] - e2['DepDate']).total_seconds())
#                         if e1['To'] == e2['From'] else None
#                     ),
#                     'Dep–Arr': (
#                         abs((e1['DepDate'] - e2['ArrDate']).total_seconds())
#                         if e1['From'] == e2['To'] else None
#                     )
#                 }
            
#                 for label, diff in time_diffs.items():
#                     if diff is not None and diff < threshold:
#                         matches.append({
#                             'Type': label,
#                             'Diff_hours': round(diff / 3600, 2),
#                             'IMO1': e1['IMO'],
#                             'Route1': (e1['From'], e1['To']),
#                             'Dep1': e1['DepDate'],
#                             'Arr1': e1['ArrDate'],
#                             'IMO2': e2['IMO'],
#                             'Route2': (e2['From'], e2['To']),
#                             'Dep2': e2['DepDate'],
#                             'Arr2': e2['ArrDate']
#                         })
#                 # # Display results
#                 # for m in matches:
#                 #     print(f"\n🟡 Match ({m['Type']}) with {m['Diff_hours']} hours gap:")
#                 #     print(f"  IMO {m['IMO1']} - {m['Route1']} | Dep: {m['Dep1']}, Arr: {m['Arr1']}")
#                 #     print(f"  IMO {m['IMO2']} - {m['Route2']} | Dep: {m['Dep2']}, Arr: {m['Arr2']}") 
#     else:
#         next
# # select one ship for testing with networkx
# tanker_9252278 = alltankers[alltankers['IMO'] == 9252278]
# tanker_9666742 = alltankers[alltankers['IMO'] == 9666742]
# c = (1,2,3,4,5,6)
# c.shift(1)
# # plot using count frequency
# G_9252278 = nx.MultiDiGraph()
# G_9522128 = nx.DiGraph()
# #G_9252278.add_nodes_from(tanker_9252278['PORTNAME'].unique())
# # egde_portname = []
# # for edge in range(len(tanker_9252278['PORTNAME'])-1):

# #     tup = tuple([tanker_9252278['PORTNAME'].iloc[edge],tanker_9252278['PORTNAME'].iloc[edge+1], 1])
# #     egde_portname.append(tup)
    
# # an alternative to  create sequential edges based on the sequential trips   
# egde_portname_9252278 = list(zip(tanker_9252278['PORTNAME'][:-1], tanker_9252278['PORTNAME'][1:]))
# G_9252278.add_edges_from(egde_portname_9252278)
# # assign attribute values for each edge
# nx.set_edge_attributes(G_9252278, values = 1, name = 'weight')
# # calculate frequency of edges
# edge_freq = Counter(G_9252278.edges())

# G_9252278 = nx.DiGraph()
# # make new graph with weighted edges calculated above
# for edge, w in edge_freq.most_common():

#     G_9252278.add_edge(*edge)
#     G_9252278.edges[*edge]['weight'] =edge_freq[*edge]
# # extract edges from new calculation
# edges = []   
# for ((u,v), value) in edge_freq.items():
#     dict1 = tuple([u,v, {'weight' : value}])
#     edges.append(dict1)

# # create a new graph with just calculated edges
# new_G = nx.MultiDiGraph()
# new_G.add_edges_from(edges)
# new_edges = new_G.edges()
# # seperate weight from the graph
# weight = [new_G[u][v]['weight'] for u,v, w in new_G.edges(data = True)]
# # create dictionary with key = edge and values = weight for ploting edge labels later
# edge_labels=dict([((u,v,),d['weight'])
#              for u,v,d in new_G.edges(data=True)])    

# # Plot new graph with weigted edges and edge labels
# plt.subplots(figsize = (16,10), dpi = 80)
# pos = nx.spring_layout(new_G)
# nx.draw(new_G, pos, with_labels= True)
# nx.draw_networkx_edges(new_G, pos = pos)
# nx.draw_networkx_edge_labels(new_G, pos, edge_labels=edge_labels, font_size=10, label_pos = 0.5)
# # Handle self-loops manually
# # Manually draw label for self-loop if needed
# for u, v, d in new_G.edges(data=True):
#     if u == v:  # self-loop
#         x, y = pos[u]
#         plt.text(x + 0.05, y + 0.05, str(d['weight']), fontsize=10, color='red')
# plt.show()

# # in case want to plot based on weight weight, have to standadize the weight

# # extract weights of edges to standadize them
# edge, weight = zip(*nx.get_edge_attributes(G_9252278, 'weight').items())

# # Normalize weights to range 1–10 (or tweak as needed)
# min_w, max_w = min(weight), max(weight)
# scaled_weights = [
#     1 + 9 * (w - min_w) / (max_w - min_w) if max_w != min_w else 5
#     for w in weight
# ]

# pos = nx.kamada_kawai_layout(G_9252278)
# plt.subplots(figsize= (16,10), dpi = 200)
# nx.draw_networkx(
#     G_9252278,
#     pos,
#     width=scaled_weights,
#     node_size=300,
#     font_size=10,
#     arrows=True, arrowsize = 20
# )

# # %% plot using Digraph but show multiple edges at the same location, except for looping
# # G = nx.DiGraph()
# # G.add_edges_from(egde_portname_9252278)
# # pos = nx.spring_layout(G, seed=5)
# # fig, ax = plt.subplots(figsize = (10,8), dpi =280)
# # nx.draw_networkx_nodes(G, pos, ax=ax, node_size = 80)
# # nx.draw_networkx_labels(G, pos, ax=ax, font_size = 8)
# # fig.savefig("1.png", bbox_inches='tight', pad_inches=0)


# # curved_edges = [edge for edge in G.edges() if (edge[1], edge[0]) in G.edges()]
# # straight_edges = list(set(G.edges()) - set(curved_edges))
# # nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
# # arc_rad = 0.25
# # nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,
# #                        connectionstyle=f'arc3, rad = {arc_rad}')


# # nx.set_edge_attributes(G, values = 1, name = 'weight')
# # nx.get_edge_attributes(G, "weight")

# # %%
# # %% plot using multidigraph without calculate the frequency of edges
# G = nx.MultiDiGraph()
# vessel_9252278 = []
# for n in range(len(egde_portname_9252278)):
#     k = tuple([egde_portname_9252278[n],
#                {'ARRVLTIME':str(tanker_9252278['ARRIVALDATE'].iloc[n+1])},
#                {'DEPDATE' : str(tanker_9252278['DEPDATE'].iloc[n+1])}])
#     vessel_9252278.append(k)
# arrts_9252278  = []   
# for n in range(len(egde_portname_9252278)):
#     k = {'ARRVLTIME':str(tanker_9252278['ARRIVALDATE'].iloc[n+1])}
    
#     arrts_9252278.append(k)
# egde_portname_9252278 = list(zip(tanker_9252278['PORTNAME'][:-1], tanker_9252278['PORTNAME'][1:]))

# for n in range(len(egde_portname_9252278)):
#     G.add_edge(egde_portname_9252278[1], arrival = str(tanker_9252278['ARRIVALDATE'].iloc[1+1]))

# list(G.edges())
# nx.set_edge_attributes(G, values = 1, name = 'weight')

# list(G.adjacency())
# pos = nx.kamada_kawai_layout(G)
# plt.figure(figsize = (12,10), dpi = 50)
# nx.draw_networkx(
#     G,
#     pos, arrowsize = 15
# )

# G.add_edges_from(egde_portname_9252278, dict(alltankers['ARRIVALDATE']))
# vessel_9252278 = []
# for n in range(len(egde_portname_9252278)):
#     u, v = egde_portname_9252278[n]
#     k = tuple([u, v,
#                {'ARRVLTIME':str(tanker_9252278['ARRIVALDATE'].iloc[n+1]), 
#                'DEPDATE' : str(tanker_9252278['DEPDATE'].iloc[n+1]),
#                'TRAVELTIME_y' : str(tanker_9252278['TRAVELTIME_y'].iloc[n+1])}])
#     vessel_9252278.append(k)
    
# G = nx.MultiDiGraph()
# G.add_edges_from(vessel_9252278)
# vessel_9252278[1]
# list(G.adjacency())
# # %% plot trips of different vessels in different color
#         # new_start_date = next_iter_day - timedelta(days = 5)
#         # next_iter_day = next_iter_day + timedelta(days = 5)
#         # if (new_start_date < last_date) & (next_iter_day < last_date):
#         #     seg_tankers = alltankers_adjusted[(
#         #         alltankers_adjusted['DepDate'] >= start_date)
#         #         & (alltankers_adjusted['DepDate'] <= next_iter_day)]
                                          
#         #     edges_and_attributes = []
#         #     for n in range(len(seg_tankers)):
#         #         info = tuple([seg_tankers['DepPort'].iloc[n], 
#         #                       seg_tankers['ArrPort'].iloc[n],
#         #                       {'DepDate' : str(seg_tankers['DepDate'].iloc[n]),
#         #                        'ArrDate' : str(seg_tankers['ArrDate'].iloc[n]),
#         #                        'TravelTime' : str(seg_tankers['TravelTime'].iloc[n]),
#         #                        'IMO': seg_tankers['IMO'].iloc[n]}])
#         #         edges_and_attributes.append(info)
#         #     # plot graph
            
#         #     # Work on smaller skill
#         #     # kamada_kawai_layout
            
#         #     Graph = nx.MultiDiGraph()
#         #     Graph.add_edges_from(edges_and_attributes)
#         #     # pos = nx.kamada_kawai_layout(Graph)
#         #     # plt.figure(figsize = (16,12), dpi = 100)
#         #     # nx.draw_networkx(
#         #     #     Graph,
#         #     #     pos, arrowsize = 15, with_labels= False, node_size = 50, font_size = 8
#         #     # )
            
#         #     imo_gr = [k['IMO'] for n,m, k in Graph.edges(data=True)]
#         #     edge_freq = Counter(Graph.edges())

            
#         #     # Get unique IMOs
#         #     unique_imos = list(set(imo_gr))
            
#         #     # Assign a unique color to each
#         #     color_list = list(mcolors.TABLEAU_COLORS.values())
#         #     color_map = {imo: color_list[i % len(color_list)] for i, imo in enumerate(unique_imos)}
            
#         #     # Map edges to their colors
#         #     edge_colors = [color_map[data['IMO']] for u, v, data in Graph.edges(data=True)]
#         #     node_clr = ['pink' if node in ports_of_interest else 'black' if node in port_of_russia
#         #                 else 'blue' for node in Graph.nodes()]
#         #     pos = nx.kamada_kawai_layout(Graph)
#         #     plt.figure(figsize = (16,12), dpi = 100)
#         #     nx.draw(Graph, pos, with_labels=True, edge_color=edge_colors, width=2, node_color = node_clr)
#         #     # Create legend
            
#         #     legend_elements = [Patch(facecolor=color_map[imo], label=f'IMO {imo}') for imo in unique_imos]
#         #     plt.legend(handles=legend_elements, title="Edge IMO", loc='upper left', bbox_to_anchor=(1, 1))
            
#         #     plt.tight_layout()
#         #     plt.show()