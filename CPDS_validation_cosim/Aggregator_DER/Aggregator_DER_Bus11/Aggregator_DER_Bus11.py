# -*- coding: utf-8 -*-
"""
Controller simulator
"""
from datetime import datetime
import cmath
import json
import math
import re
import os
import sys
import numpy as np
import helics as h
import opendssdirect as dss
from opendssdirect.utils import run_command
import time

import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dir_to_Controller = os.getcwd()
print(f"python_dir={os.getcwd()}")
dir_to_project = os.path.join(dir_to_Controller,'..','..',)
dir_to_Algorithm = os.path.join(dir_to_project, 'Module_files')
sys.path.append(dir_to_Algorithm)
import vsm_utils

################# read in config file ###########################################
print("reading json")
filename = "Aggregator_DER_setup.json"
with open(filename) as f:
    data = json.loads(f.read())
    federate_name = data["name"]
    feeder_name = federate_name.split('_')[-1]
    total_time = data["total_time"]
    simulation_step_time = data["simulation_step_time"]
    num_DER = data["num_DER"]
    scale_P = data["scale_P"]
    scale_Q = data["scale_Q"]
    DER_rating_MW_prescale = data["DER_rating_MW_prescale"]
    Total_DER_ratings = data["Total_DER_ratings"]


print("reading json okay!")

################### create the federate info object ##################

PUBLICATIONS = {}
SUBSCRIPTIONS = {}

#time.sleep(3)

# define federate information (configuration)
fedinfo = h.helicsCreateFederateInfo() # initialization
h.helicsFederateInfoSetCoreName(fedinfo, f"{federate_name}") # define name
h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq") # core type:
h.helicsFederateInfoSetCoreInitString(fedinfo, "--federates=1") # define number of federate connect to this core
h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, 0.01)

# create the value federate: this type of federate is used for exchange electrical information
fed = h.helicsCreateCombinationFederate(f"{federate_name}", fedinfo)
print(f"{federate_name}: Combination federate created", flush=True)

# register endpoints
print('endpoint name:')
print(federate_name+"_rACE")

epid_rACE = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_rACE", "") # receive ACE from TransmissionSim/send DER AGC response
# epid_feeder = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_feeder", "") # receive vsm/send pvd1_power to feeder
# epid_power = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_power", "") # send DER pmax to transmission/receive pvd1 power output from transmission
epid_sAGC = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_sAGC", "") # receive ACE from TransmissionSim/send DER AGC response



h.helicsFederateEnterExecutingMode(fed)
print(f"{federate_name}: Entering execution mode", flush=True)
################### Now start executation ###################
vsm = None
allbusmagpu_base = None

#scale = 0.478*100/0.970657 # 23.5
#DER_caps = np.ones(num_DER)*900/scale_P # in kw

current_time = 0
PVD_AGC_response = np.zeros(num_DER)
#num_nodes=95
index_DERs = np.array([12, 25, 26, 27, 28, 29, 30, 92, 93, 94])
#index_DERs = np.arange(num_nodes-10, num_nodes, 1) # attach DER to the last 10 nodes
pvd1_power = np.ones(num_DER)*500/scale_P # initialize pvd1_power

time_list = []
DER_local_limit_from_vsm_list = []
total_DER_power_limit = 0
AGC_participation_feeder = DER_rating_MW_prescale/Total_DER_ratings # feeder level DER_ratings/total_DER_rating; put total_DER_rating in Json
# could be two level of participation
AGC_participation_each = np.array(num_DER*[1/num_DER])
AGC_participation_each = np.array([0.1,0.2,0.3,0.4])
AGC_participation_feeder = 0.1
AGC_participation = AGC_participation_feeder * AGC_participation_each

for request_time in np.arange(0, total_time, simulation_step_time): # simulation_step_time is from json
    while current_time < request_time:

        current_time = h.helicsFederateRequestTime(fed, request_time)
    time_list.append(current_time)
    print(f"current_time={current_time}")

    # initialized ACE = 0
    if current_time == 0:
        ACE = 0

    # check message from transmission: get ACE
    while h.helicsEndpointHasMessage(epid_rACE):
        m = h.helicsEndpointGetMessage(epid_rACE)
        message = json.loads(m.data)
        ACE = message
        print(f"Received message at time {current_time}: ACE {ACE} 100 MW")

    # check message from Distribution: get vsm
    # while h.helicsEndpointHasMessage(epid_feeder):
    #     m = h.helicsEndpointGetMessage(epid_feeder)
    #     message = json.loads(m.data)
    #     vsm = np.array(message[0])
    #     allbusmagpu_base = np.array(message[1])
    #     #print(f"Received message at time {current_time}: vsm {vsm[index_DERs,:]}")
    #
    # # check message from transmission: get DER_power output
    # while h.helicsEndpointHasMessage(epid_power):
    #     m = h.helicsEndpointGetMessage(epid_power)
    #     message = json.loads(m.data)
    #     pvd1_power = np.array(message)*1e5/scale_P
    #     print(f"Received message at time {current_time}: pvd1_power {message} 100 MW")
    # to_endpoint = feeder_name+'_ep'
    # h.helicsEndpointSendBytesTo(epid_feeder, json.dumps(pvd1_power.tolist()), to_endpoint)
    # print(f"sending pvd1_power={pvd1_power} in kw to feeder {feeder_name}.")

    ##### use ACE calculate PVD AGC 1 ###########
    #PVD_AGC_response = PVD_AGC_response - ACE*simulation_step_time * AGC_participation * KI # this is vector
    PVD_AGC_response =  ACE* AGC_participation
    #print(f"PVD_AGC_response={PVD_AGC_response}")
    #PVD_AGC_response_total = np.sum(PVD_AGC_response)
    PVD_AGC_response_feeder_name = np.append(feeder_name, PVD_AGC_response)

    if current_time % 4 <= 1e-6: # it was 4 sec
        # h.helicsEndpointSendBytesTo(epid_sAGC, json.dumps(PVD_AGC_response.tolist()), f"TransmissionSim_DER_AGC_{feeder_name}")
        # print(f"sending PVD_AGC_response for {feeder_name}={PVD_AGC_response}")
        # print("to endpoint name:")
        # print(f"TransmissionSim_DER_AGC_{feeder_name}")
        h.helicsEndpointSendBytesTo(epid_sAGC, json.dumps(PVD_AGC_response_feeder_name.tolist()), f"TransmissionSim_rAGC")
        print(f"sending PVD_AGC_response for {feeder_name}={PVD_AGC_response_feeder_name}")
        print("to endpoint name:")
        print(f"TransmissionSim_DER_AGC_{feeder_name}")

    # if current_time % 10 <= 1e-6: # can be 60s interval get_DER_local_limit
    #     if vsm is not None and allbusmagpu_base is not None:
    #         # perform optimization
    #         DER_headroom_df, total_DER_headroom_limit = vsm_utils.get_DER_local_limit(vsm, DER_caps-pvd1_power, allbusmagpu_base)
    #         # get output headroom
    #         print(f"DER_limit_df={DER_headroom_df}")
    #         total_DER_power_limit = pvd1_power + np.array(DER_headroom_df.solution_value)
    #         # this DER_limit_df*scale should be sent to Transmission
    #         h.helicsEndpointSendBytesTo(epid_power, json.dumps((total_DER_power_limit*scale_P/1e5).tolist()), "TransmissionSim_DER_HEADROOM_{feeder_name}")
    #         print(f"sending total_DER_power_limit={total_DER_power_limit}")
    #
    # DER_local_limit_from_vsm_list.append(total_DER_power_limit)

# DER_local_limit_from_vsm_df = pd.DataFrame()
# DER_local_limit_from_vsm_df['time_s']=time_list
# DER_local_limit_from_vsm_df['DER_local_limit_from_vsm'] =DER_local_limit_from_vsm_list
#
# now = datetime.now() # record the time
# append_time = now.strftime("%m_%d_%Y_%H_%M_%S")
# dir_to_results = os.path.join(dir_to_Controller, "simulation_results")
# DER_local_limit_from_vsm_df.to_csv(dir_to_results+'\\DER_local_limit_from_vsm_df_'+append_time+'.csv')


h.helicsFederateFinalize(fed)

h.helicsFederateFree(fed)
h.helicsCloseLibrary()
print("Federate finalized")

