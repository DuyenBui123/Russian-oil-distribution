# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 02:23:26 2025

@author: Duyen
""" 
import os
cwd = os.getcwd()
#os.chdir('D:\\Dropbox\\Duyen\\University\\Master\\Year 2\\Internship\\')
processes = os.cpu_count() - 2
from itertools import islice
import sys
import collections
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import tqdm
import itertools
from datetime import datetime, timedelta
import networkx as nx
from Code import data_processing as pr
from Code import data_preprocessing as pp
from Code import routefinding as rf
import numpy as np
import pandas as pd
import psutil
import joblib
import multiprocess
from  multiprocess import Pool
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

alltankers_adjusted = pd.read_csv('./processing/pr_inter_input/RU_oil_tankers_data.csv',
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

# %% Port selections for different regions

country_of_interest = ['China', 'India', 'Turkey', 'United Arab Emirates', 'Singapore',
                       'Saudi Arabia', 'Korea (South)', 'Brazil', 'Kazakhstan', 'Russia', 'Netherlands']
ru_country = ['Russia']
eu_countries = [
    "Netherlands","Sweden",  "Belgium", "Greece", "Italy", "France", "Spain",
    "Germany", "Finland", "Poland", "Denmark", "Portugal", "Romania", "Lithuania",
    "Ireland", "Malta", "Cyprus", "Bulgaria", "Croatia", "Slovenia", "Estonia",
    "Latvia"]
NL = ['Netherlands']
port_of_interest = {}
for ctry in country_of_interest:
    ctry_ports = alltankers_adjusted[
        alltankers_adjusted['Country'] == ctry]['DepPort'].unique()
    port_of_interest[ctry] = list(ctry_ports)

# Select RU port

ru_country = ['Russia']
port_of_russia = pr.extract_ports_based_countries(alltankers_adjusted, ru_country)


eu_ports = pr.extract_ports_based_countries(alltankers_adjusted, eu_countries)
# Select the Dutch ports

NL_ports =  pr.extract_ports_based_countries(alltankers_adjusted, NL)
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
# extract betweeness centrality of ports for each country
bwtcentr_ports = []
for ctr_of_int in country_of_interest:
    filter_value = {port: btwcentr[port] for port in port_of_interest[ctr_of_int]}
    # Get top 2 keys with highest values
    top_2_keys = sorted(filter_value, key=filter_value.get, reverse=True)[:2]
    bwtcentr_ports.append(top_2_keys)
bwtcentr_ports = [port for sublist in bwtcentr_ports for port in sublist]

# extract ports of NL and RU with the highest betweeness centrality
start_RU_port = list(set(bwtcentr_ports) & set(port_of_russia))
end_port = list(set(bwtcentr_ports) & set(NL_ports))
# remove RU and NL port from the port lists of highest betweeness centrality
for port in start_RU_port:
    bwtcentr_ports.remove(port)
for port in end_port:
    bwtcentr_ports.remove(port)

#start_RU_port = ['Novorossiysk']

# %% start extracting route
# tot_nr_port = (3,4,5,6,7,8)
# iterat_time = (8,7,6,5,4,3)
from memory_profiler import memory_usage
tot_nr_port = (7, 8)
iterat_time = (4, 3)

start_time = time.time()
upperbound_time = 24*30#float('inf')
lowerbound_time = 0
win_time_slide = 1
strike = 'None'
loop = False
loop_type = 'country'


port_of_interest = bwtcentr_ports
# outputpath1 = './processing/pr_inter_output/loop_potential_routes_nrRU_2_nrtotport6.joblib'
# outputpath2 = './processing/pr_inter_output/loop_potential_routes_nrRU_2_nrtotport4.joblib'
# # Round 1

# # start iterating
# if __name__ == '__main__':
#     rf.route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
#                       win_time_slide, 1, 
#                       strike, 6, outputpath1, Graph_whole_dataset, 
#                       port_of_russia, eu_ports, bwtcentr_ports,
#                       alltankers_adjusted, ru_country,  loop, loop_type)


# %%
zip_iter_and_nrport = list(zip(tot_nr_port,iterat_time))
peak_memory = 0

def monitor_memory(interval=1):
    global peak_memory
    process = psutil.Process(os.getpid())
    while True:
        try:
            mem = process.memory_info().rss / (1024**2)  # in MiB
            peak_memory = max(peak_memory, mem)
            time.sleep(interval)
        except psutil.NoSuchProcess:
            break  # process ended

def func(zip_iter_and_nrport):
    # for tup in zip_iter_and_nrport:
        nr_of_port, iterat_time = zip_iter_and_nrport

        start_time = time.time()
        upperbound_time = 24*30#float('inf')
        lowerbound_time = 0
        win_time_slide = 1
        strike = 'None'
        loop = False
        loop_type = 'country'
        
        port_of_interest = bwtcentr_ports
        outputpath = f'./processing/pr_inter_output/loop_potential_routes_nrRU_{len(start_RU_port)}_nrtotport{nr_of_port}.joblib'
        # Round 1
        
        # start iterating
        
        final_route_RU_to_NL = rf.route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
                          win_time_slide, iterat_time, 
                          strike, nr_of_port, outputpath, Graph_whole_dataset, 
                          port_of_russia, eu_ports, port_of_interest,
                          alltankers_adjusted, ru_country,  loop, loop_type)
        return final_route_RU_to_NL
time_and_men = []
runtime_path = f'./processing/pr_inter_output/timeandmem_RU_{len(start_RU_port)}.joblib'
import psutil, os, time, threading
t = threading.Thread(target=monitor_memory, daemon=True)
t.start()
for zipl in zip_iter_and_nrport:
    mem_usage = memory_usage((func, (zipl,)), interval = 60)
    time_and_men.append(mem_usage)
    joblib.dump(time_and_men, runtime_path)

print(f"Peak memory before crash: {peak_memory:.2f} MiB")    
peak = max(time_and_men[1])

# Get system RAM (in MiB)
total_ram = psutil.virtual_memory().total / (1024**2)

percent_used = (peak_memory / total_ram) * 100

print(f"Peak memory usage: {peak_memory:.2f} MiB "
      f"({percent_used:.2f}% of system RAM)")

a = joblib.load(runtime_path)
# # load
     = joblib.load('./processing/pr_inter_input/timeandmem_RU_2.joblib')
# a = joblib.load('./processing/pr_inter_output/iter4.joblib')

# route_RU_int_NL_matched_imoNr,trip_freq_dict, country_seq, port_sequence = pr.route_seq_matched_nrimo_par(
#     final_route_RU_to_NL, alltankers_adjusted, 4, 3,  False, oiltype = 'crude oil', loop_type = 'port')
# aa = []
# for df in a:
#     v = alltankers_adjusted.loc[df]
#     aa.append(v)

# # %% check memory
# # # import sys
# # from pympler.asizeof import asizeof
# # #sys.getsizeof(filtered_final_route_RU_to_NL_old)
# # #asizeof(filtered_final_route_RU_to_NL_old)
# # sys.getsizeof(filtered_final_route_RU_to_NL)
# # asizeof(filtered_final_route_RU_to_NL)

# #%% extract routes further based on requirement dsuch as nr of ports and imo


    
# # final_route_RU_to_NL = []
# # for lst in filtered_final_route_RU_to_NL:
# #     for sublst in lst:

# #             final_route_RU_to_NL.append(sublst)

# # select for the total number of IMO
# # check duplicates in the routes
# seen = set()
# duplicates = []

# for sublist in final_route_RU_to_NL:
#     tup = tuple(sublist)
#     if tup in seen:
#         duplicates.append(sublist)
#     else:
#         seen.add(tup)
        
        




            
            
# a = sorted(port_sequence.items(), key = lambda item: item[1], reverse=True) 
# # extract freq of trip based on the result of freq of routes
# key_list = [] 
# for tup in a[0:10]:
#     key,val = tup
#     key_list.append(key)
# key_freq = []
# for key in key_list:
    
#     aa = trip_freq_dict[key]
#     key_freq.append(aa)



        
# # %% Analyze and extract values      
# # # add new value in the exisiting                
# # # delete row used from the original and update        

# #             match_df = route_RU_int_NL_matched_imoNr_mer[route_RU_int_NL_matched_imoNr_mer[]]
        
# #             port_list = tuple(port_list)
# #             if port_list not in list(port_sequence.keys()):
# #                 port_sequence[port_list] =  1
# #             else:
# #                 port_sequence[port_list] = port_sequence.get(port_list) + 1

# # def freq_of_port_seq(route_RU_int_NL_matched_imoNr):
# #     port_sequence = {}
# #     for df in route_RU_int_NL_matched_imoNr:
    
# #         port_list = df['DepPort'].tolist()
# #         port_list.append(df.iloc[-1]['ArrPort'])
# #         port_list = tuple(port_list)
# #         if port_list not in list(port_sequence.keys()):
# #             port_sequence[port_list] =  1
# #         else:
# #             port_sequence[port_list] = port_sequence.get(port_list) + 1
# #     return port_sequence

# # ## select all df with the highest freq
# # sorted_keys = sorted(port_sequence.items(), key=lambda item: item[1], reverse=True)

# # # Get the first key and convert it to a list
# # first_key = list(sorted_keys[0][0])
# # # extract dataframe has the higest freq

# # list_matched_df = []
# # for df in route_RU_int_NL_matched_imoNr:
# #     dep_ports = df['DepPort'].tolist()

# #     if dep_ports == first_key[0:len(first_key)-1]:
# #         list_matched_df.append(df)
    
# # # calculate the total time for a complete route from RU to NL
# # time_trvl_in_mth = []
# # for df in route_RU_int_NL_matched_imoNr:
# #     df = route_RU_int_NL_matched_imoNr[1]
# #     travel_dur = sum(df['TravelTime'])

# #     #travel_dur = abs(df['DepDate'].iloc[0] - df['ArrDate'].iloc[-1])
# #     travel_dur = round((travel_dur.days * scnd_in_day 
# #                         + travel_dur.seconds)/(24*3600*30.47), 0)
# #     time_trvl_in_mth.append(travel_dur)

# # time_trvl_uniq = list(set(time_trvl_in_mth))
# # time_trvl_dict = {item: time_trvl_in_mth.count(item) for item in time_trvl_uniq}

# # # calculate travel time and not matching time
# # time_trvl_in_mth = []
# # for df in route_RU_int_NL_matched_imoNr:

# #     #travel_dur = abs(df['DepDate'].iloc[0] - df['ArrDate'].iloc[-1])
# #     travel_dur = round(sum((df['TravelTime'].dt.days * scnd_in_day 
# #                         + df['TravelTime'].dt.seconds))/(24*3600*30.47), 0)
# #     time_trvl_in_mth.append(travel_dur)

# # time_trvl_uniq = list(set(time_trvl_in_mth))
# # time_trvl_dict = {item: time_trvl_in_mth.count(item) for item in time_trvl_uniq}
# # # plot
# # plt.bar(range(len(time_trvl_dict)), list(time_trvl_dict.values()), align = 'center')
# # plt.xticks(range(len(time_trvl_dict)), list(time_trvl_dict.keys()))
# # plt.ylabel('Nr. of routes')
# # plt.xlabel('Travel time (in month)')
# # plt.title('Travel duration (not include matching time)')
# # plt.show() # ADD total port and imo
# # # check the routes that have travel duration time smaller than a certain number
# # route_w_certain_trvl_dur = []
# # for df in route_RU_int_NL_matched_imoNr:

# #     travel_dur = abs(df['DepDate'].iloc[0] - df['ArrDate'].iloc[-1])
# #     travel_dur = round((travel_dur.days * scnd_in_day
# #                         + travel_dur.seconds)/(24*3600*30.47), 0)
# #     if travel_dur <=3 and travel_dur >0: # ADD certain month for checking
# #         route_w_certain_trvl_dur.append(df)
        
# # #  calculate the frequency of imo involved in the potential RU oil transportation
# # # in a given nr of ports and nr of imo
# # route_RU_int_NL_matched_imoNr_merge = pd.concat(route_RU_int_NL_matched_imoNr, 
# #                                                 ignore_index= True)
  
# # imo_array = np.concatenate([df['IMO'].unique() for df in route_RU_int_NL_matched_imoNr]).tolist()
# # imo_uniq = list(set(imo_array))
# # imo_dict = {item: imo_array.count(item) for item in imo_uniq}
# # check_route_w_pp_imo = [9383950]
# # check_route_w_pp_imo = [9230505]
# # route_w_pp_imo = [df for df in route_RU_int_NL_matched_imoNr if df['IMO'].isin(check_route_w_pp_imo).any()]
# # # check if there is any same set of imo occurs
# # set_imo = [tuple(df['IMO'].unique()) for df in route_RU_int_NL_matched_imoNr]
# # set_imo_uniq = np.unique(set_imo)
# # set_imo_dict = {item: set_imo.count(item) for item in imo_uniq}
# # counter = collections.Counter(set_imo)
# # # count round trip
# # route_back_to_RU = []
# # for df in route_RU_int_NL_matched_imoNr:
    
# #     arr_ctry = df['Arr_Country']
# #     RU_count =  int((arr_ctry == 'Russia').sum())
# #     route_back_to_RU.append(RU_count)
# # count_visit_RU_pr_NL = collections.Counter(route_back_to_RU)

# # count_df = pd.DataFrame.from_dict(count_visit_RU_pr_NL, orient='index', columns=['Number of Routes'])
# # count_df.sort_index(inplace=True)  # Sort by number of RU visits

# # # Plot
# # count_df.plot(kind='bar', legend=False)
# # plt.title('Nr of time a trip revisit Russia before arriving in the Netherlands')
# # plt.xlabel('Nr of time revisiting Russia ')
# # plt.ylabel('Number of Routes')
# # plt.xticks(rotation=0)
# # plt.tight_layout()
# # plt.show()
# # %% def main function
# def route_finding(start_RU_port, end_port, lowerbound_time, upperbound_time,
#                   win_time_slide, iterat_time, 
#                   strike, tot_port_nr, loop, loop_type):
#     """
# Function finding all possible routes from givens start ports and end ports. 
# The result of extracted routes can vary based on the prior determined rules such as:

# + No EU or RU ports in the first intermediate ports with a start port from a RU port

# + If a route includes an EU port before arriving to an NL port, the same IMO is used to 
# transport oil from e.g. a non EU port->EU port->NL port

# Extra rules can be set based on the preference of users such as:
# + Connection time between two trips
# + Circular routes 
# + Ship type
# + Ports of interest

# Args:
#     start_RU_port (list of strings): a list of RU port names where a route begins
    
#     end_port (list of strings): a list of NL/or any port names where a route ends
    
#     lowerbound_time (int): in hours, the lower bound of the time interval used
#     to find matching IMOs at a shared port
    
#     upperbound_time (int) : in hours, the upper bound of the time interval used
#     to find matching IMO at a shared port
    
#     win_time_slide (int): used to extend the interval time (including lower- and
#                         and upper-bound). This parameter allows to extend a constant
#                         time distance based on the initial defined interval.
#                         If this parameter is deactivated, it functions as a iteration
#                         counter for a while-loop
                        
#     strike (int): in hours. Similar to the win_time_slide, but it allows to extend
#     an inconsistent time distance
    
#     iterat_time (int): the total number of iteration time will allow to update 
#     and exent the lower- and upper-bound of the time interval used to find new 
#     matching IMOs at a shared port
    
#     tot_port_nr (int): Total number of ports included in a compete route
    
#     loop (bool): a boolean input with True: remove all circular routes in a route
#     with False keeping all types of routes
    
#     loop_type (string): there are two types of loops: port and country.
#     A route can contain a port/or country more than once in its sequence


# Returns:
#     final_route_RU_to_NL (list of list): a list of index lists. Each index list
#     contains a sequence of a full route. To view all data of the routes, using 
#     the original data table (alltanker_adjusted)
# """
    
#     strike = 24*7
    
#     win_time_slide = 1
#     iterat_time = 2
#     routes_comb_w_multi_win_sld = []
#     start_iter = 0
#     tot_nr_port = 3
#     n = start_iter
#     m = tot_nr_port
#     scnd_in_day = 1*24*60*60
#     processes = os.cpu_count() - 2
#     while win_time_slide <= iterat_time:
#         # improve time window
#         if strike == 'None':      
            
#             upperbound_time = upperbound_time*win_time_slide
#             lowerbound_time = lowerbound_time
#         else:
#             upperbound_time = upperbound_time
#             lowerbound_time = lowerbound_time
#         # start to run model
#         # iter
        
#         nbs_edges_RU_cmb = []
#         nbs_edges_RU = []
#         filtered_final_route_RU_to_NL = []
#         if len(start_RU_port) == 1:
#             nbs_edges_RU = list(set(list(Graph_whole_dataset.out_edges(start_RU_port,))))
            
#         else:
#             for ruport in start_RU_port:
#                 nbs_edges = list(set(list(Graph_whole_dataset.out_edges(ruport,))))
#                 nbs_edges_RU_cmb.append(nbs_edges)
        
#         # unlist nbs edges
#         if len(nbs_edges_RU_cmb) > 1:
#             for lst in nbs_edges_RU_cmb:
#                 for sublst in lst:
#                     nbs_edges_RU.append(sublst)
        
#         # loop through all neighbours of Novorossiysk
        
#     # Parallel computation code for finding route at the first intermediate port
#         # if len(nbs_edges_RU) > processes:
#         #     processes = processes
#         # else:
#         #     processes = 1
#         # chunk_size = len(nbs_edges_RU)//processes
#         # chunks = [nbs_edges_RU[i:i + chunk_size] for i in range(0, len(nbs_edges_RU), chunk_size)]
        
        
#         #     # prepare argument tuples
#         # args = [(chunk, port_of_russia,
#         # eu_ports, alltankers_adjusted, 
#         # scnd_in_day, lowerbound_time, upperbound_time) for chunk in chunks]
        
#         # with Pool(processes=processes) as pool:
#         #     track_route_fr_RU_to_2ndPort_and_connected_IMO = pool.starmap(
#         #         pr.find_matched_imo_at_1stshared_port,
#         #         args
#         #     )
#         # track_route_fr_RU_to_2ndPort_and_connected_IMO = list(
#         #     itertools.chain.from_iterable(
#         #         track_route_fr_RU_to_2ndPort_and_connected_IMO))
    
#     # code without parallel computation   
#         track_route_fr_RU_to_2ndPort_and_connected_IMO = pr.find_matched_imo_at_1stshared_port(
#             nbs_edges_RU, port_of_russia,
#             eu_ports, alltankers_adjusted, 
#             scnd_in_day, lowerbound_time, upperbound_time)
    
        
#         # extract route from RU-hotpot-NL and number of IMO difference            
#         route_RU_1int_NL = []
#         route_RU_1int_other = []
#         # extract route from RU-aport-NL
#         for df in track_route_fr_RU_to_2ndPort_and_connected_IMO:
        
#             info_shared_port = alltankers_adjusted.loc[df]
        
#             if (info_shared_port['ArrPort'].isin(bwtcentr_w_NLport)).any():
#                 if (info_shared_port['DepPort'].isin(bwtcentr_ports)).any():
                    
#                     route_RU_1int_NL.append(df)
                    
#             else:
#                 route_RU_1int_other.append(df)
#         #* save final route from RU to NL
#         filtered_final_route_RU_to_NL.append(route_RU_1int_NL)      
    
#         # delete not necessary variable 
#         del  track_route_fr_RU_to_2ndPort_and_connected_IMO
#         # iteration calculation
#         n = n+1
#         # Round 2:
#         # get nbs of the next port
#         route_RU_to_NL = route_RU_1int_other
    
#         if len(route_RU_to_NL) == 0:
#             raise ValueError('The total number of possible routes should be'
#                              ' larger than 0. It is possible that the'
#                              ' time interval is too small, no many available'
#                              ' IMO meets the requirements')
    
        
    
#         def f(x):
#             return x*x
#         if __name__ == '__main__':
#             __spec__ = None
#             with Pool(5) as p:
#                print (p.map(f, [1, 2, 3]))         
               
        
#     # phase 2
#         while n < m:
#                  #* save final routes from RU to NL
#             if len(route_RU_to_NL) >processes:
#                 processes = processes
#             else:
#                 processes = 1
#             chunk_size = len(route_RU_to_NL)//processes
#             chunks = [route_RU_to_NL[i:i + chunk_size] for i in range(0, len(route_RU_to_NL), chunk_size)]
            
            
#                 # prepare argument tuples
#             args = [(chunk, upperbound_time, lowerbound_time, alltankers_adjusted,
#                                                            scnd_in_day, True, 'country') for chunk in chunks]
            
#             with Pool(processes=processes) as pool:
#                 track_route_fr_RU_to_NL = pool.starmap(
#                     pr.find_matched_imo_at_shared_port_noloop_par,
#                     args
#                 )
#             del args
#             if len(track_route_fr_RU_to_NL) == 0:
#                 raise ValueError('The total number of possible routes after filtering'
#                                  ' based on the pre-defined conditions should be larger than 0')
            
#             # pool = multiprocess.Pool(processes = processes)
        
#             # track_route_fr_RU_to_NL = pool.map(pr.find_matched_imo_at_shared_port_noloop_par, chunks)   
        
#             track_route_fr_RU_to_NL = [lst for lst in track_route_fr_RU_to_NL if len(lst)>0]
#             track_route_fr_RU_to_NL = list(itertools.chain.from_iterable(track_route_fr_RU_to_NL))
            
    
        
#             route_RU_int_NL, route_RU_int_other = pr.extract_route_RU_to_NL_and_others(
#                 track_route_fr_RU_to_NL, alltankers_adjusted,
#                 bwtcentr_w_NLport,
#                 bwtcentr_ports)
         
                
                
#             # next port(eu) to NL
            
#                  #* save final routes from RU to NL
#             if len(route_RU_int_NL) >processes:
#                 processes = processes
#             else:
#                 processes = 1
                
#             if len(route_RU_int_NL) == 0:
#                 raise ValueError('The total number of possible routes from RU to NL'
#                                  ' has to be greater than 0. It is possible that the'
#                                  ' time interval is too small, not many available'
#                                  ' IMOs meet the requirements')
                 
#             chunk_size = len(route_RU_int_NL)//processes
#             print('chunk_size of iter', n, chunk_size, 'en length of the whole route', len(route_RU_int_NL))
#             chunks = [route_RU_int_NL[i:i + chunk_size] for i in range(0, len(route_RU_int_NL), chunk_size)]
#             pool = multiprocess.Pool(processes = processes)
#             # prepare argument tuples
#             args = [(chunk, alltankers_adjusted, ru_country, port_of_russia) for chunk in chunks]
#             with Pool(processes=processes) as pool:
#                 route_RU_int_NL_filtered_v1 = pool.starmap(pr.filter1, args)
                
#             if len(route_RU_int_NL_filtered_v1) == 0:
#                 raise ValueError('The total number of possible routes from RU to NL'
#                                  ' has to be greater than 0. It is possible that'
#                                  ' the time interval is too small, not many available'
#                                  ' IMOs meets the requirements. No routes match the'
#                                  ' requirements, after filtering all'
#                                  ' routes containing a RU port in the sequence going'
#                                  ' direct to NL')
              
        
#             route_RU_int_NL_filtered_v1 = [lst for lst in route_RU_int_NL_filtered_v1 if len(lst)>0]
#             route_RU_int_NL_filtered_v1 = list(itertools.chain.from_iterable(route_RU_int_NL_filtered_v1))
#             del chunk_size, chunks, args
#             if len(route_RU_int_NL_filtered_v1) >processes:
#                 processes = processes
#             else:
#                 processes = 1
#             chunk_size = len(route_RU_int_NL_filtered_v1)//processes
#             chunks = [route_RU_int_NL_filtered_v1[i:i + chunk_size] for i in range(0, len(route_RU_int_NL_filtered_v1), chunk_size)]
#             args = [(chunk,  alltankers_adjusted, eu_ports) for chunk in chunks]
#             with Pool(processes=processes) as pool:
#                 route_RU_int_NL_filtered_v2 = pool.starmap(pr.filter2, args)
        
#             del chunk_size, chunks, args
            
#             if len(route_RU_int_NL_filtered_v2) == 0:
#                 raise ValueError(' The total number of possible routes from RU to NL'
#                                  ' has to be greater than 0. It is possible that'
#                                  ' the defined time interval is too small, not many available'
#                                  ' IMOs meets the requirements. No routes match the'
#                                  ' requirements, after filtering all'
#                                  ' routes that contain different IMOs from an EU country'
#                                  ' to NL')
        
            
#             route_RU_int_NL_filtered_v2 = [lst for lst in route_RU_int_NL_filtered_v2 if len(lst)>0]
#             route_RU_int_NL_filtered_v2 = list(itertools.chain.from_iterable(route_RU_int_NL_filtered_v2))
            
#             filtered_final_route_RU_to_NL.append(route_RU_int_NL_filtered_v2)
#             if len(route_RU_int_NL_filtered_v2) == 0:
#                 # displaying the warning
#                 warnings.warn(f'The total number of possible routes from RU to NL'
#                               f' after all filter equal 0 when total number of port'
#                               f' reach {tot_nr_port + 2}')
#             # update list of routes for the next round iteration
#             route_RU_to_NL = route_RU_int_other
#             del route_RU_int_NL_filtered_v2, route_RU_int_NL_filtered_v1, route_RU_int_NL
    
#             if len(route_RU_int_other) == 0:
#                 raise ValueError('The total number of possible routes for the'
#                                  ' next iteration should be larger than 0')
    
#             #del chunk_size, chunks
#             print('iter:', n)
#             print('used mem:', psutil.virtual_memory().percent)
#             run_time = (time.time() - start_time)
#             print('run time:', run_time)
#             n = n+1
#         if strike == 'None':
#             lowerbound_time = upperbound_time
#             upperbound_time = 24*7
#             win_time_slide = win_time_slide+1
#         else:
#            lowerbound_time = upperbound_time
#            upperbound_time = upperbound_time + strike
#            win_time_slide = win_time_slide+1
           
#         routes_comb_w_multi_win_sld.append(filtered_final_route_RU_to_NL)
#         n = start_iter
#         m = tot_nr_port
#     # combine all results from different time window slide

#     if len(routes_comb_w_multi_win_sld) >1:
#         for lst in range(len(routes_comb_w_multi_win_sld)):
#             final_route_RU_to_NL = list(itertools.chain.from_iterable(routes_comb_w_multi_win_sld))
#             list_depth = pr.depth(final_route_RU_to_NL)-2
#             for lst in range(list_depth):
#                 final_route_RU_to_NL = list(itertools.chain.from_iterable(final_route_RU_to_NL))
            
#     else:
#         list_depth = pr.depth(routes_comb_w_multi_win_sld)-2
#         final_route_RU_to_NL = routes_comb_w_multi_win_sld
#         for lst in range(list_depth):
#             routes_comb_w_multi_win_sld = list(itertools.chain.from_iterable(routes_comb_w_multi_win_sld))

#     return final_route_RU_to_NL