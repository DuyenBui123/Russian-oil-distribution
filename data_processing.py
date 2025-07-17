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