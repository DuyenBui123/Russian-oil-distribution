# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 17:43:35 2025

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


path = r'D:\Dropbox\Duyen\University\Master\Year 2\Internship\processing\pr_inter_output'

# Prefix patterns mapped to variable names
patterns = {
    "Performance_loop_nrRU_1_time1w": "dfs_loop_1w1RU",
    "Performance_loop_nrRU_1_time2w": "dfs_loop_2w1RU",
    "Performance_loop_nrRU_1_time3w": "dfs_loop_3w1RU",
    "Performance_loop_nrRU_1_time4w": "dfs_loop_4w1RU"
}
patterns_noloop = {
    "Performance_noloop_nrRU_1_time1w": "dfs_loop_1w1RU",
    "Performance_noloop_nrRU_1_time2w": "dfs_loop_2w1RU",
    "Performance_noloop_nrRU_1_time3w": "dfs_loop_3w1RU",
    "Performance_noloop_nrRU_1_time4w": "dfs_loop_4w1RU"
}

patterns_dff_input = {
    "Performance_noloop_nrRU_1_time4w": "dfs_loop_1w1RU",
    "Performance_noloop_nrRU_2_time4w": "dfs_loop_2w1RU",
    "Performance_noloop_nrRU_3_time4w": "dfs_loop_3w1RU",
    "Performance_noloop_nrRU_4_time4w": "dfs_loop_4w1RU"
}

mem_patterns = {
    "proc_log_file_loop_RU_1_1w": "dfs_loop_1w1RU",
    "proc_log_file_loop_RU_1_2w": "dfs_loop_2w1RU",
    "proc_log_file_loop_RU_1_3w": "dfs_loop_3w1RU",
    "proc_log_file_loop_RU_1_4w": "dfs_loop_4w1RU"
}
mem_patterns_noloop = {
    "proc_log_file_noloop_RU_1_1w": "dfs_loop_1w1RU",
    "proc_log_file_noloop_RU_1_2w": "dfs_loop_2w1RU",
    "proc_log_file_noloop_RU_1_3w": "dfs_loop_3w1RU",
    "proc_log_file_noloop_RU_1_4w": "dfs_loop_4w1RU"
}
mem_diff_input = {
    "proc_log_file_noloop_RU_1_4w": "dfs_loop_1w1RU",
    "proc_log_file_noloop_RU_2_4w": "dfs_loop_2w1RU",
    "proc_log_file_noloop_RU_3_4w": "dfs_loop_3w1RU",
    "proc_log_file_noloop_RU_4_4w": "dfs_loop_4w1RU"
}
# Read all files by pattern into a dictionary of DataFrame dicts
dfs_dict = {}

for prefix, var_name in patterns.items():
    matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
    dfs_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}
    
dfs_mem_dict = {}
for prefix, var_name in mem_patterns.items():
    matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
    dfs_mem_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}

dfs_mem_noloop_dict = {}
for prefix, var_name in mem_patterns_noloop.items():
    matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
    dfs_mem_noloop_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}    

dfs_mem_noloop_dff_input_dict = {}
for prefix, var_name in mem_diff_input.items():
    matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
    dfs_mem_noloop_dff_input_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}   
    
dfs_noloop_dict = {}
for prefix, var_name in patterns_noloop.items():
    matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
    dfs_noloop_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}

dfs_noloop_dff_input_dict = {}
for prefix, var_name in patterns_dff_input.items():
    matched_files = [f for f in os.listdir(path) if f.startswith(prefix)]
    dfs_noloop_dff_input_dict[var_name] = {f: pd.read_csv(os.path.join(path, f)) for f in matched_files}
    
runtime_colect = []
for key, val  in dfs_dict.items():
    dict_2 = dfs_dict[key]
    runtime_pertime = []
    for key2, val2 in dict_2.items():
        runtime = val2.iloc[-1,3]
        runtime_pertime.append(runtime)
    runtime_colect.append(runtime_pertime)

max_len = [len(lst) for lst in runtime_colect]
max_len = max(max_len)
for lst in runtime_colect:
    leng =len(lst)
    max_len.append(leng)
    
# convert seconds to hours
for lst in range(len(runtime_colect)):
    runtime_colect[lst] = [ round(x/60,0) for x in runtime_colect[lst]]
# Pad each list with np.nan to match max_len
runtime_colect = [lst + [np.nan] * (max_len - len(lst)) for lst in runtime_colect]

max_mem_collect = []   
for key, val  in dfs_mem_dict.items():
    dict_2 = dfs_mem_dict[key]
    maxmem_pertime = []
    for key2, val2 in dict_2.items():
        mem = val2['Process Memory (MB)'].max()
        maxmem_pertime.append(mem)
    max_mem_collect.append(maxmem_pertime)
        
max_mem_collect = [lst + [np.nan] * (max_len - len(lst)) for lst in max_mem_collect]

runtime_noloop_collect = []

for key, val  in dfs_noloop_dict.items():
    dict_2 = dfs_noloop_dict[key]
    runtime_pertime = []
    for key2, val2 in dict_2.items():
        runtime = val2.iloc[-1,3]
        runtime_pertime.append(runtime)
    runtime_noloop_collect.append(runtime_pertime)
        
runtime_noloop_collect = [lst + [np.nan] * (max_len - len(lst)) for lst in runtime_noloop_collect]


# convert seconds to hours
for lst in range(len(runtime_noloop_collect)):
    runtime_noloop_collect[lst] = [ round(x/60,0) for x in runtime_noloop_collect[lst]]
# Pad each list with np.nan to match max_len
runtime_noloop_collect = [lst + [np.nan] * (max_len - len(lst)) for lst in runtime_noloop_collect]

max_mem_noloop_collect = []   
for key, val  in dfs_mem_noloop_dict.items():
    dict_2 = dfs_mem_noloop_dict[key]
    maxmem_pertime = []
    for key2, val2 in dict_2.items():
        mem = val2['Process Memory (MB)'].max()
        maxmem_pertime.append(mem)
    max_mem_noloop_collect.append(maxmem_pertime)
        
max_mem_noloop_collect = [lst + [np.nan] * (max_len - len(lst)) for lst in max_mem_noloop_collect]
# Replace last row
#max_mem_collect[-1] = [np.nan, np.nan, 4602.0703125, np.nan, np.nan, np.nan]

runtime_noloop_dff_input_collect = []

for key, val  in dfs_noloop_dff_input_dict.items():
    dict_2 = dfs_noloop_dff_input_dict[key]
    runtime_pertime = []
    for key2, val2 in dict_2.items():
        runtime = val2.iloc[-1,3]
        runtime_pertime.append(runtime)
    runtime_noloop_dff_input_collect.append(runtime_pertime)
        
runtime_noloop_dff_input_collect = [lst + [np.nan] * (max_len - len(lst)) for lst in runtime_noloop_dff_input_collect]


# convert seconds to hours
for lst in range(len(runtime_noloop_dff_input_collect)):
    runtime_noloop_dff_input_collect[lst] = [ round(x/60,0) for x in runtime_noloop_dff_input_collect[lst]]
    
max_mem_noloop_dff_input_collect = []   
for key, val  in dfs_mem_noloop_dff_input_dict.items():
    dict_2 = dfs_mem_noloop_dff_input_dict[key]
    maxmem_pertime = []
    for key2, val2 in dict_2.items():
        mem = val2['Process Memory (MB)'].max()
        maxmem_pertime.append(mem)
    max_mem_noloop_dff_input_collect.append(maxmem_pertime)
        
max_mem_noloop_dff_input_collect = [lst + [np.nan] * (max_len - len(lst)) for lst in max_mem_noloop_dff_input_collect]

# x-axis starting from 3
from matplotlib.ticker import MultipleLocator


x = np.arange(3, 3 + max_len)

plt.figure(figsize=(8,5))
plt.plot(x, runtime_colect[0], marker='o', label='1w')
plt.plot(x, runtime_colect[1], marker='o', label='2w')
plt.plot(x, runtime_colect[2], marker='o', label='3w')
plt.plot(x, runtime_colect[3], marker='o', label='4w')
#plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
plt.xlabel("Total number of ports", fontsize=20)
plt.ylabel("Runtime (hours)", fontsize=20)
plt.title("Runtime Comparison", fontsize=30)
plt.legend(fontsize=20)
plt.grid(True)

# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Set x-axis tick interval to 1
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.show()

x = np.arange(3, 3 + max_len)

plt.figure(figsize=(8,5))
plt.plot(x, max_mem_collect[0], marker='o', label='1w')
plt.plot(x, max_mem_collect[1], marker='o', label='2w')
plt.plot(x, max_mem_collect[2], marker='o', label='3w')
plt.plot(x, max_mem_collect[3], marker='o', label='4w')
#plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
plt.xlabel("Total number of ports", fontsize = 20)
plt.ylabel("Used Memory (MiB)", fontsize = 20)
plt.title("Used Memory Comparison", fontsize = 20)
plt.legend(fontsize=20)
plt.grid(True)
# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Set x-axis tick interval to 1
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.show()

plt.figure(figsize=(8,5))
plt.plot(x, runtime_noloop_collect[0], marker='o', label='1w')
plt.plot(x, runtime_noloop_collect[1], marker='o', label='2w')
plt.plot(x, runtime_noloop_collect[2], marker='o', label='3w')
plt.plot(x, runtime_noloop_collect[3], marker='o', label='4w')
#plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
plt.xlabel("Total number of ports", fontsize = 20)
plt.ylabel("Run time (minutes)", fontsize = 20)
plt.title("Run time Comparison for the case of no circular routes", fontsize = 20)
plt.legend(fontsize=20)
plt.grid(True)
# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Set x-axis tick interval to 1
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.show()


plt.figure(figsize=(8,5))
plt.plot(x, max_mem_noloop_collect[0], marker='o', label='1w')
plt.plot(x, max_mem_noloop_collect[1], marker='o', label='2w')
plt.plot(x, max_mem_noloop_collect[2], marker='o', label='3w')
plt.plot(x, max_mem_noloop_collect[3], marker='o', label='4w')
#plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
plt.xlabel("Total number of ports", fontsize = 20)
plt.ylabel("Used Memory (MiB)", fontsize = 20)
plt.title("Used Memory Comparison for the case of no circular routes", fontsize = 20)
plt.legend(fontsize=20)
plt.grid(True)
# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Set x-axis tick interval to 1
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.show()



plt.figure(figsize=(8,5))
plt.plot(x, runtime_noloop_dff_input_collect[0], marker='o', label='1w')
plt.plot(x, runtime_noloop_dff_input_collect[1], marker='o', label='2w')
plt.plot(x, runtime_noloop_dff_input_collect[2], marker='o', label='3w')
plt.plot(x, runtime_noloop_dff_input_collect[3], marker='o', label='4w')
#plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
plt.xlabel("Total number of ports", fontsize = 20)
plt.ylabel("Run time (minutes)", fontsize = 20)
plt.title("Run time Comparison for the case of different input with no circular routes", fontsize = 20)
plt.legend(fontsize=20)
plt.grid(True)
# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Set x-axis tick interval to 1
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.show()


plt.figure(figsize=(8,5))
plt.plot(x, max_mem_noloop_dff_input_collect[0], marker='o', label='1w')
plt.plot(x, max_mem_noloop_dff_input_collect[1], marker='o', label='2w')
plt.plot(x, max_mem_noloop_dff_input_collect[2], marker='o', label='3w')
plt.plot(x, max_mem_noloop_dff_input_collect[3], marker='o', label='4w')
#plt.plot(x, runtime_1m1RU_loop_hours, marker='o', label='1m1RU loop')
plt.xlabel("Total number of ports", fontsize = 20)
plt.ylabel("Used Memory (MiB)", fontsize = 20)
plt.title("Used Memory Comparison for the case of different input with no circular routes", fontsize = 20)
plt.legend(fontsize=20)
plt.grid(True)
# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Set x-axis tick interval to 1
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

plt.show()