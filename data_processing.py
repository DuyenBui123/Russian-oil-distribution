# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:44:55 2025

@author: Duyen

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools as it
import networkx as nx
import datetime
import collections

def potential_IMO_at_shared_port(track_route_fr_RU_to_2ndPort_and_connected_IMO, 
                                 route_from_RUport_to_its_nb, start_RU_port, 
                                 diff_IMO_at_2ndPort, 
                                 scnd_in_day, low_t_time, up_t_time,start_IMO,cons_IMO,cons_IMO_nr):
    
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
    return track_route_fr_RU_to_2ndPort_and_connected_IMO, start_IMO, cons_IMO, cons_IMO_nr




def potential_IMO_at_cons_shared_port(tracking_filtered_snd_oiltransshipment_imo_list, 
                                      row_in_imo_list, 
                                      tracking_each_nb_next_oiltrans_connect_IMO, 
                                 route_from_RUport_to_its_nb, start_RU_port, 
                                 diff_IMO_at_2ndPort, 
                                 scnd_in_day, low_t_time, up_t_time,start_IMO_v1,cons_IMO_v1,cons_IMO_nr_v1):
    start_IMO_v2 = {} # start from a specific RU port. Expect a dict of dict. with the
    # first layer contain RU port names and its connected IMO info in general.
    # The secondlayer key: nb name and its attributes
    cons_IMO_v2 = {} # the next second port met conditions (no EU port) and contain a connected IMO
    cons_IMO_nr_v2 = [] # contain a connected IMO 
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
    start_IMO_v1[start_RU_port] = cons_IMO_v1
    return tracking_each_nb_next_oiltrans_connect_IMO, start_IMO_v1, cons_IMO_v1, cons_IMO_nr_v1


def cons_transfer(start_IMO, port_of_russia,eu_ports,
                  track_route_fr_RU_to_2ndPort_and_connected_IMO,
                  alltankers_adjusted,
                  start_RU_port,
                  scnd_in_day, low_t_time, up_t_time
                  ):
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
        route_from_RUport_to_its_nb = tracking_filtered_snd_oiltransshipment_imo_list[row_in_imo_list].iloc[[-1]]
        arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
        # extract all IMO available at the arrival port of the first trip from RU
        diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(arr_port)]
        # for each nb, calculate time gap between IMO from RU to 2nd port and 
        # the IMO available at the 2nd port
        tracking_all_nbs_next_oiltrans_connect_IMO, start_IMO_v1, cons_IMO_v1, cons_IMO_nr_v1 = potential_IMO_at_cons_shared_port(
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
    return start_IMO, cons_IMO, cons_IMO_nr, track_route_fr_RU_to_2ndPort_and_connected_IMO

def path_freq_of_a_IMO_fr_A_to_B(freq_IMO_RUport_to_2ndport, 
                                alltankers_adjusted):
    # frequency occurance of each IMO in each path
    path_frq_RU_to_hotspot = pd.DataFrame(columns = ['IMO','RU_China',
                                                     'RU_Ind', 'RU_Tur',
                                                     'RU_Kaz',
                                                     'RU_Egp',
                                                     'RU_Braz','RU_Sing',
                                                     'RU_Sau_Ara', 'RU_Korea(S)',
                                                     'RU_Malay',
                                                     'RU_UAE',
                                                     'RU_Azer'])
    for row in range(len(freq_IMO_RUport_to_2ndport)):
        RU_1stport = freq_IMO_RUport_to_2ndport[row:row+1]
        imo_cntry = alltankers_adjusted[alltankers_adjusted['IMO'] == RU_1stport.loc[row, 'IMO']]['Country']
        cons_pair_cntry = zip(imo_cntry, imo_cntry[1:])
    
        cons_pair_cntry_count = collections.Counter(list(cons_pair_cntry))
        path_frq_RU_to_hotspot.loc[row, 'IMO'] = RU_1stport.loc[row, 'IMO']
        if ('Russia', 'India') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Ind'] = cons_pair_cntry_count[('Russia', 'India')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Ind'] = 0
        if ('Russia', 'Turkey') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Tur'] = cons_pair_cntry_count[('Russia', 'Turkey')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Tur'] = 0
        if ('Russia', 'Kazakhstan') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Kaz'] = cons_pair_cntry_count[('Russia', 'Kazakhstan')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Kaz'] = 0
        if ('Russia', 'Brazil') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Braz'] = cons_pair_cntry_count[('Russia', 'Brazil')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Braz'] = 0
        if ('Russia', 'Egypt') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Egp'] = cons_pair_cntry_count[('Russia', 'Egypt')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Egp'] = 0
        if ('Russia', 'China') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_China'] = cons_pair_cntry_count[('Russia', 'China')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_China'] = 0        
        if ('Russia', 'Singapore') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Sing'] = cons_pair_cntry_count[('Russia', 'Singapore')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Sing'] = 0 
        if ('Russia', 'United Arab Emirates') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_UAE'] = cons_pair_cntry_count[('Russia', 'United Arab Emirates')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_UAE'] = 0 
        if ('Russia', 'Saudi Arabia') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Sau_Ara'] = cons_pair_cntry_count[('Russia', 'Saudi Arabia')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Sau_Ara'] = 0 
        if ('Russia','Korea (South)') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Korea(S)'] = cons_pair_cntry_count[('Russia','Korea (South)')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Korea(S)'] = 0 
        if ('Russia','Malaysia') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Malay'] = cons_pair_cntry_count[('Russia','Malaysia')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Malay'] = 0 
        if ('Russia','Azerbaijan') in cons_pair_cntry_count:
            path_frq_RU_to_hotspot.loc[row, 'RU_Azer'] = cons_pair_cntry_count[('Russia','Azerbaijan')]
        else:
            path_frq_RU_to_hotspot.loc[row, 'RU_Azer'] = 0 

        
    # sum the frequency of each row
    path_frq_RU_to_hotspot['Sum'] = path_frq_RU_to_hotspot.loc[:,'RU_China':'RU_Azer'].sum(axis=1)
    return path_frq_RU_to_hotspot
def path_freq_of_a_IMO_fr_B_to_NL(freq_IMO_2ndport_to_NLport,
                                  alltankers_adjusted):
    path_frq_2ndhotspot_to_NL = pd.DataFrame(columns = ['IMO','China_NL',
                                                     'Ind_NL', 'Tur_NL',
                                                     'Kaz_NL','Egp_NL',
                                                     'Braz_NL','Sing_NL',
                                                     'Sau_Ara_NL', 'Korea(S)_NL',
                                                     'Malay_NL','UAE_NL',
                                                     'Azer_NL'])
    for row in range(len(freq_IMO_2ndport_to_NLport)):
        sndport_NL = freq_IMO_2ndport_to_NLport[row:row+1]
        imo_cntry = alltankers_adjusted[alltankers_adjusted['IMO'] == sndport_NL.loc[row, 'IMO']]
        cons_pair_cntry = zip(imo_cntry['Country'], imo_cntry['Arr_Country'])
    
        cons_pair_cntry_count = collections.Counter(list(cons_pair_cntry))
        path_frq_2ndhotspot_to_NL.loc[row, 'IMO'] = sndport_NL.loc[row, 'IMO']
        if ('India', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Ind_NL'] = cons_pair_cntry_count[('India','Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Ind_NL'] = 0
        if ('Turkey','Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Tur_NL'] = cons_pair_cntry_count[('Turkey','Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Tur_NL'] = 0
        if ('Egypt', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Egp_NL'] = cons_pair_cntry_count[('Egypt', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Egp_NL'] = 0
        if ('China', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'China_NL'] = cons_pair_cntry_count[('China', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'China_NL'] = 0        
        if ('Kazakhstan', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Kaz_NL'] = cons_pair_cntry_count[('Kazakhstan', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Kaz_NL'] = 0 
        if ('United Arab Emirates', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'UAE_NL'] = cons_pair_cntry_count[('United Arab Emirates', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'UAE_NL'] = 0 
        if ('Brazil', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Braz_NL'] = cons_pair_cntry_count[('Brazil', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Braz_NL'] = 0 
        if ('Singapore', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Sing_NL'] = cons_pair_cntry_count[('Singapore', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Sing_NL'] = 0 
        if ('Saudi Arabia', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Sau_Ara_NL'] = cons_pair_cntry_count[('Saudi Arabia', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Sau_Ara_NL'] = 0 
        if ('Korea (South)', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Korea(S)_NL'] = cons_pair_cntry_count[('Korea (South)', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Korea(S)_NL'] = 0
        if ('Malaysia', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Malay_NL'] = cons_pair_cntry_count[('Malaysia', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Malay_NL'] = 0       
        if ('Azerbaijan', 'Netherlands') in cons_pair_cntry_count:
            path_frq_2ndhotspot_to_NL.loc[row, 'Azer_NL'] = cons_pair_cntry_count[('Azerbaijan', 'Netherlands')]
        else:
            path_frq_2ndhotspot_to_NL.loc[row, 'Azer_NL'] = 0
    # sum the frequency of each row
    path_frq_2ndhotspot_to_NL['Sum'] = path_frq_2ndhotspot_to_NL.loc[:,'China_NL':'Azer_NL'].sum(axis=1)
    return path_frq_2ndhotspot_to_NL
def path_likelyhood(paths_frB_toC):
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
    
    # Convert to DataFrame for better viewing or plotting
    a_path_likelyhood = pd.DataFrame(
        [(dep, arr, pct) for (dep, arr), pct in percentage_dict.items()],
        columns=["DepPort", "ArrPort", "Percentage"]
    ).sort_values(by="Percentage", ascending=False)
    return a_path_likelyhood
def freq_of_port_seq(route_RU_int_NL_matched_imoNr):
    port_sequence = {}
    for df in route_RU_int_NL_matched_imoNr:
    
        port_list = df['DepPort'].tolist()
        port_list.append(df.iloc[-1]['ArrPort'])
        port_list = tuple(port_list)
        if port_list not in list(port_sequence.keys()):
            port_sequence[port_list] =  1
        else:
            port_sequence[port_list] = port_sequence.get(port_list) + 1
    return port_sequence

def freq_of_country_seq(route_RU_int_NL_matched_imoNr):
    country_sequence = {}
    for df in route_RU_int_NL_matched_imoNr:
    
        ctry_list = df['Country'].tolist()
        ctry_list.append(df.iloc[-1]['Arr_Country'])
        ctry_list = tuple(ctry_list)
        if ctry_list not in list(country_sequence.keys()):
            country_sequence[ctry_list] =  1
        else:
            country_sequence[ctry_list] = country_sequence.get(ctry_list) + 1
    return country_sequence
def route_seq_matched_nrimo(route_RU_int_NL,
                            req_nr_port, req_nr_imo):
    filtered_dfs = list(filter(lambda df: df.shape[0] == req_nr_port, 
                               route_RU_int_NL))
    route_RU_int_NL_matched_imoNr = []
    for df in filtered_dfs:
        info_shared_port = df
        if len(info_shared_port['IMO'].unique()) == req_nr_imo: # fill in TOTAL NR. OF PORT ALLOWED
            route_RU_int_NL_matched_imoNr.append(info_shared_port)
    if len(route_RU_int_NL_matched_imoNr) == 0:
        print('No matched results')
    return route_RU_int_NL_matched_imoNr

def extract_route_RU_to_NL_and_others(track_route_fr_RU_to_NL,
                                      bwtcentr_w_NLport,
                                      bwtcentr_ports):
    # extract port NL
    route_RU_int_NL = []
    route_RU_int_other = []
    # extract route from RU-aport-NL
    # check hotpot
    for df in track_route_fr_RU_to_NL:
    
        info_shared_port = df.iloc[-1]
        if (info_shared_port['ArrPort'] in (bwtcentr_w_NLport)):
            if np.isin(df['DepPort'].unique(), (bwtcentr_ports)).any():
                
                route_RU_int_NL.append(df)
                
        else:
            route_RU_int_other.append(df)
    return route_RU_int_NL, route_RU_int_other

def find_matched_imo_at_1stshared_port(nbs_edges_RU, port_of_russia,
                                       eu_ports, alltankers_adjusted,
                                       scnd_in_day, low_t_time, up_t_time):
    track_route_fr_RU_to_2ndPort_and_connected_IMO = []
    for edge in nbs_edges_RU:
    
        # extract route from a RU port to its neighbout
        start_at_RU_port = edge[0]
        if ((edge[1] in (eu_ports)) or (edge[1] in port_of_russia)):
            next
        
        else: # forloop check the first stop in EU or RU
    
           
            route_from_RUport_to_its_nb = alltankers_adjusted[alltankers_adjusted['DepPort'].isin([start_at_RU_port]) &
                                                          alltankers_adjusted['ArrPort'].isin([edge[1]])]
            arr_port = list(route_from_RUport_to_its_nb['ArrPort'])
            # extract all IMO available at the arrival port of the first trip from RU
            diff_IMO_at_2ndPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
                arr_port)]
            
            
            if len(diff_IMO_at_2ndPort)>=1:       
                    
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
                            
                        else:
                            next
    
    
            else: # for matched IMO at shared ports
                next
    
    return track_route_fr_RU_to_2ndPort_and_connected_IMO
def find_matched_imo_at_shared_port(route_RU_to_NL,alltankers_adjusted, 
                                   df,
                                  scnd_in_day,
                                  low_t_time,
                                  up_t_time
                                  ):
    track_route_fr_RU_to_NL = []
    for df in route_RU_to_NL:

        arr_port = [df['ArrPort'].iloc[-1]]
        # extract all IMO available at the arrival port of the first trip from RU
        diff_IMO_at_sharedPort = alltankers_adjusted[alltankers_adjusted['DepPort'].isin(
            arr_port)]
        
        
        if len(diff_IMO_at_sharedPort)>=1:       
                

                
                # arrive time of IMO travel from RU to its nb
                arr_time_of_IMO_fr_RU_to_2ndPort = df.iloc[-1,
                                                      df.columns.get_loc('ArrDate')]

                test_test_ = pd.DataFrame(columns = df.columns)
                # loop through all IMO availabel in a nb
                for row_dep_port in range(len(diff_IMO_at_sharedPort)):
                    # departure time of an IMO
                    dep_time_of_IMO_avai_at_sharedPort = diff_IMO_at_sharedPort.iloc[row_dep_port,
                                                      diff_IMO_at_sharedPort.columns.get_loc('DepDate')]
                    dep_port_per_row = diff_IMO_at_sharedPort.iloc[row_dep_port,
                                                      diff_IMO_at_sharedPort.columns.get_loc('DepPort')]
                    IMO_avai_at_2ndPort = diff_IMO_at_sharedPort.iloc[row_dep_port:
                                                      row_dep_port+1]
            
                    # time different between IMO from RU and IMO availabe at its nb
                    time_gap = dep_time_of_IMO_avai_at_sharedPort - arr_time_of_IMO_fr_RU_to_2ndPort
                    time_gap_hr = (time_gap.days*scnd_in_day + time_gap.seconds)/(60*60)
                    # base on threshold assump to take for oil transshipment. if met 
                    # the threshold condition, save attributes of that IMO. Otherwise move to the next IMO of IMO available list
                    if (np.sign(time_gap_hr) == 1) & ((abs(time_gap_hr)>= low_t_time) & (abs(time_gap_hr) < up_t_time)):
                        #print(f'time gap after the condition {time_gap_hr}')
            
                        
                        df_merg = pd.concat([df, IMO_avai_at_2ndPort])
                        if len(df_merg) > len(df):
                            
                            track_route_fr_RU_to_NL.append(df_merg)
                        
                    else:
                        next
    return track_route_fr_RU_to_NL

