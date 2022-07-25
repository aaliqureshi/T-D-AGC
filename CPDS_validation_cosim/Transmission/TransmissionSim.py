# -*- coding: utf-8 -*-
"""
Transmission simulator using Andes:
To check andes model properties
print(ss.PQ.doc())
"""
from datetime import datetime
import cmath
import json
import math
import re
import os
import sys
import time
import numpy as np
import helics as h
import opendssdirect as odd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import andes
import pickle

start_time = time.time()

##################### First define a broker for total cosim ###########################
path_to_TransmissionSim = os.getcwd()
path_to_results = os.path.join(os.getcwd(), "..", "Results_figures")
path_to_transmission_xlsx_folder = os.path.join(os.getcwd(), "..", "Network_models", "Transmission")

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

################# reading in config file ###########################################
print("reading json")
filename = "TransmissionSim.json"
with open(filename) as f:
    data = json.loads(f.read())
    federate_name = data["name"]
    file_name = data["file_name"]
    total_time = data["total_time"]
    DER_AGC_flag = data["DER_AGC_flag"]
    tgov_AGC_flag = data["tgov_AGC_flag"]
    KP, KI = data["AGC_controller_kp_ki"]
    subscriptions = data["subscriptions"]
    publications = data["publications"]
    simulation_step_time = data["simulation_step_time"]
    num_feeder = data["num_feeder"]
    num_federate = data["num_federate"]
    feeder_list = data["feeder_list"]
    num_DER = data["num_DER"]
    num_DER_list = list(num_DER.values())
    penetration_scenario = data["penetration_scenario"]
    ACE_bias = data["ACE_bias"]

print("reading json okay!")

# AGC PI controller KP KI
# KP = 0.2
# KI = 0.2
#generation_drop_time = 40
# DER_AGC_flag = True
# tgov_AGC_flag = False
########## Create federate ###############

# define federate information (configuration)
fedinfo = h.helicsCreateFederateInfo()

fedinitstring = "--federates=1" # this string is for defining the core
h.helicsFederateInfoSetCoreName(fedinfo, f"{federate_name}") # define name
h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq") # core type
h.helicsFederateInfoSetCoreInitString(fedinfo, fedinitstring) # define number fo federate and tell the federate it is the main broker
h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, 0.01)
#h.helicsFederateInfoSetLoggingLevel(fedinfo, 5)  # from loglevel in json

# create value federate based on federate info: this type of federate is used for exchange electrical information
fed = h.helicsCreateCombinationFederate(f"{federate_name}", fedinfo)
print(f"{federate_name}: Combination federate created", flush=True)

# ###################  initialize transmission systems #####################################
andes.main.config_logger(stream_level=20)



path_to_transmission_xlsx = os.path.join(path_to_transmission_xlsx_folder, file_name)

ss = andes.run(path_to_transmission_xlsx, setup=False, no_output=True)


# this is to add fault
# ss.add("Toggler", dict(model='SynGen', dev="GENROU_3", t=10.0))

################################## Start changing scenario ###############################

# read in base values
# load_total = ss.PQ.as_df().p0.sum()
# pv_total = ss.PV.as_df().p0.iloc[9:].sum()
# print(f"pv_total for baseline scenario (10% penetration) = {pv_total}")
# pv_base = list(ss.PV.as_df().p0.iloc[9:])
# thermal_base = list(ss.PV.as_df().p0.iloc[:9])
# pv_idx = list(ss.PV.as_df().idx)
# slack_p0 = ss.Slack.as_df().p0.iloc[0]
# thermal_base_with_slack = thermal_base + [slack_p0]

# adjust pv output p0
# for ind, elem in enumerate(pv_idx[9:]):
#     ss.PV.set(src='p0', idx=elem, attr='v', value=pv_base[ind] * penetration_scenario * 10)
# pv_increase = sum(pv_base) * penetration_scenario * 10 - sum(pv_base)
# thermal_decrease = pv_increase
# thermal_decrease_percent = thermal_decrease / sum(thermal_base_with_slack)
# thermal_adjust_to = np.array(thermal_base_with_slack) * (1 - thermal_decrease_percent)
# for ind, elem in enumerate(pv_idx[:9]):
#     ss.PV.set(src='p0', idx=elem, attr='v', value=thermal_adjust_to[ind])
# # adjust slack bus
# ss.Slack.set(src='p0', idx=10, attr='v', value=thermal_adjust_to[-1])
# print(f"ss.Slack.p0.v = {ss.Slack.p0.v}")
# print(f"ss.PV.p0.v = {ss.PV.p0.v}")
#
# #adjust pvd1 Sn
# shed_buses = ss.PVD1.bus.v
# pv_shed_idx = ss.PVD1.find_idx(keys='bus', values=shed_buses)
# #len(pv_shed_idx)
# pvd1_sn_base = ss.PVD1.as_df().Sn
# pvd1_sn_adjust_to = pvd1_sn_base*penetration_scenario*10
# i=0
# for ind in range(1,len(pv_shed_idx)+1):
#     ss.PVD1.set(src='Sn', idx=ind, attr='v', value = pvd1_sn_adjust_to[i])
#     i=i+1
# # adjust gov vmin for control
# tgov1n_idx = list(ss.TGOV1N.as_df().idx)
# for ind, elem in enumerate(tgov1n_idx):
#     ss.TGOV1N.set(src='VMIN', idx=elem, attr='v', value=0.1)
# print(ss.TGOV1N.VMIN.v)
# adjust ACE bias parameter
#ss.ACEc.set(src='bias', idx=1, attr='v', value=ACE_bias)


################################## adding Generation loss ###############################


#ss.add("Toggler", dict(model='GENROU', dev="GENROU_1", t=generation_drop_time))


#########################################################################################################
ss.setup()
# use constant power model for PQ (we will come back to this later)
ss.PQ.config.p2p = 1
ss.PQ.config.q2q = 0
ss.PQ.config.p2z = 0
ss.PQ.config.q2z = 1 # can change to 1 when FIDVR

#ss.PVD1.config.plim=1

# turn off under-voltage PQ-to-Z conversion
ss.PQ.pq2z = 1

#perform initial power flow for dynamic simulation initialization
ss.PFlow.run()
voltage0 = ss.Bus.v.v
#time domain simulation initialization
ss.TDS.init()

#first run 0.1 to make sure everything runs well
#ss.TDS.config.tf = 0.01

# start the 0.1 second dynamic simulation
#ss.TDS.run()
ss.dae.ts.y
shed_buses = ss.PQ.bus.v
pq_shed_idx = ss.PQ.find_idx(keys='bus', values=shed_buses)

# define turbine gov AGC settings: find index and set value
tgov_idx = list(ss.TurbineGov._idx2model.keys())
paux0_new = np.zeros(len(tgov_idx))# - ss.ACEc.ace.v*1/len(tg_idx)

ss.TurbineGov.set(src='paux0',idx=tgov_idx, attr='v', value=paux0_new)

PVD_names = ss.PVD1.idx.v
PVD_AGC_idx = ss.PVD1.find_idx(keys='idx', values=PVD_names)

PVD_AGC_response = ss.PVD1.get(src='Pext', idx=PVD_AGC_idx, attr='v')


# record the initial system load
pq_p = ss.PQ.get(src='Ppf', idx=pq_shed_idx, attr='v')
# record DER output: TODO
#ss.dae.ts.get_data([ss.PVD1.Ipout_y],a=[1,])*ss.dae.ts.get_data([ss.PVD1.v],a=[1,])*100
pvd1_current = ss.PVD1.Ipout_y.v
pvd1_voltage = ss.PVD1.v.v
pvd1_power = pvd1_current*pvd1_voltage # in 100 MW
# get the DER definition
DER_caps = ss.PVD1.as_df().Sn/100  # in 100 mw

# Sbase = 100 MW
# dae: differential–algebraic equation
# get the load time series randomly choose
consistent_random_object = np.random.RandomState(123)
load_timeseries = consistent_random_object.normal(1, 0.002, int(total_time)) # used to be 0.02
#Load_timeseries_df = pd.DataFrame(Load_timeseries)
# register the publication
PUBLICATIONS = {}
SUBSCRIPTIONS = {}
for k, v in publications.items():
    pub = h.helicsFederateRegisterGlobalTypePublication(fed, v['topic'], v['type'], "") # register to helics
    PUBLICATIONS[k] = pub # store the HELICS object

for k, v in subscriptions.items():
    sub = h.helicsFederateRegisterSubscription(fed, v['topic'], "")
    SUBSCRIPTIONS[k] = sub  # store the HELICS object



epid_ACE = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_ACE", "") # for ACE sending and receiving from thermal


# EPID_DER_AGC_R = {}
# for bus, prop in publications.items():
#     epid = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+f"_DER_AGC_Bus{bus}", "")
#     EPID_DER_AGC_R[bus] = epid
#     print('to_endpoint name')
#     print(federate_name + f"_DER_AGC_bus{bus}")

epid_rAGC = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+f"_rAGC", "")


# EPID_DER_HEADROOM_R = {}
# for bus, prop in publications.items():
#     epid = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+f"_DER_HEADROOM_bus{bus}", "")
#     EPID_DER_HEADROOM_R[bus] = epid

#EPID_AGG_DER_AGC_R = # should be a list: list of aggregator
# EPID_AGG_DER_HEADROOM = # should be a list
#
# epid1 = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_ep1", "") # for thermal
# epid2 = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_ep2", "")
# epid3 = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_ep3", "") #
# epid4 = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_ep4", "") # receive DER pmax from 34Bus
# epid5 = h.helicsFederateRegisterGlobalEndpoint(fed, federate_name+"_ep5", "") # receive DER pmax from 8500Node


# filter_trans_id = h.helicsFederateRegisterFilter(fed, h.HELICS_FILTER_TYPE_DELAY, "filter1")
# h.helicsFilterAddDestinationTarget(filter_trans_id, federate_name+"_ep")
# h.helicsFilterSet(filter_trans_id, "delay", 0.1)

############### Entering Execution Mode ########################
h.helicsFederateEnterExecutingMode(fed)
print(f"{federate_name}: Entering execution mode", flush=True)

print(f"===========Start simulation==================")
mppt=1
#
consistent_random_object_mppt = np.random.RandomState(687)
mppt_timeseries = consistent_random_object_mppt.normal(1, 0, [int(total_time), DER_caps.shape[0]]) # was 0.8, 0.05
mppt_current_all = []
#mppt_current = np.ones(20)
current_time = 0
voltage = voltage0
PVD_vsm_max_all = []
#PVD_vsm_max = np.ones(20)*0.009
pvd1_power_all = []
pvd1_qpower_all = []
#pvd1_power = np.ones(20)*0.005

time_list = []
ACE_raw_integral = 0
ACE = 0
ACE_raw = None
paux0_new = None
PVD_AGC_response = None
AGC_participation_DER = 0.1

AGC_participation_tgov = 0.9/len(tgov_idx)*np.ones(len(tgov_idx))

received_power_array=[]


#total_num_DER = int(40*num_feeder)
total_num_DER = ss.PVD1.as_df().shape[0] # int(sum(list(num_DER.values())))
print(f"total_num_DER={total_num_DER}")
for request_time in np.arange(0, total_time, simulation_step_time): # current time is end of simulation of andes
    while current_time < request_time:
        current_time = h.helicsFederateRequestTime(fed, request_time)
    time_list.append(current_time)
    print("=============================")
    print(f"request_time={request_time}")

    # save results object
    ob = (ss.dae.ts.t, ss.dae.ts.get_data([ss.Bus.v],a=[0,3,8,]))
    save_object(ob, 'ob.pk')
    print("saving ss to object.")

    pvd1_qcurrent = ss.PVD1.Iqout_y.v

    # ss.PVD1.Iqout_y.v
    pvd1_voltage = ss.PVD1.v.v
    pvd1_qpower = pvd1_qcurrent * pvd1_voltage * 100  #
    print(f"pvd1_qpower = {pvd1_qpower} at time {current_time}")
    pvd1_qpower_all.append(pvd1_qpower)
    #ob_pvd1_q = (ss.dae.ts.t, pvd1_qpower)

    save_object(pvd1_qpower_all, 'ob2.pk')

    # set power for all loads
    pq_p_new = pq_p * load_timeseries[int(current_time)]
    ss.PQ.set(src='Ppf', idx=pq_shed_idx, attr='v', value=pq_p_new)
    # here set the power from distribution (bus index=4)
    for key, sub in SUBSCRIPTIONS.items():
        if subscriptions[key]['value']=='Powers': # power meaning it is for distribution systems
            #if current_time % subscriptions[key]['sub_interval']<=1e-6:
            val = h.helicsInputGetComplex(sub) # val is P, Q
            bus_name = subscriptions[key]['element_name']
            bus_PQ_index = int(subscriptions[key]['bus_PQ_index'])-1

            P, Q = val # Xeq/Q
            received_power_array.append(P)
            print("Received active power from {} at time {}: {} kw".format(key, current_time, P))
            print("Received reactive power from {} at time {}: {} kvar".format(key, current_time, Q))
            # convert to pu
            P = P/1e5
            Q = Q/1e5

            if current_time < 1:
                print("manually set up first step load bus P and Q")
                P = ss.PQ.p0.v[int(bus_PQ_index)]
                Q = ss.PQ.q0.v[int(bus_PQ_index)]
                #
                #Xeq = ss.PQ.Xeq.v[int(bus_PQ_index)]
            print(f"bus_name={bus_name}")
            #Xeq_bus = Q/voltage[int(float(bus_name)-1)]**2
            #Xeq_bus = Xeq

            #print(f'Xeq_bus={Xeq_bus}')
            ss.PQ.set(src='Ppf', idx=pq_shed_idx[int(bus_PQ_index)], attr='v', value=P) # set load as P
            ss.PQ.set(src='Qpf', idx=pq_shed_idx[int(bus_PQ_index)], attr='v', value=Q)
            #ss.PQ.set(src='Xeq', idx=pq_shed_idx[int(bus_PQ_index)], attr='v', value=Xeq_bus) # TODO

            #print(f"total_bus_load = {ss.PQ.p0.v}")

    paux0_new = None
    # while h.helicsEndpointHasMessage(epid_ACE): # epid1 from thermal control
    #     m = h.helicsEndpointGetMessage(epid_ACE)
    #     message = json.loads(m.data)
    #     paux0_new = message
    #     #print(f"type(message)={type(message)}")
    #     print(f"Received paux0_new = {paux0_new} from thermal_controller at time {current_time}.")

    PVD_AGC_response = np.array([])
    # for bus, epid in EPID_DER_AGC_R.items():
    #     while h.helicsEndpointHasMessage(epid):
    #         m = h.helicsEndpointGetMessage(epid)
    #         message = json.loads(m.data)
    #         PVD_AGC_response_feeder = message
    #         print(f"Received PVD_AGC_response = {PVD_AGC_response_feeder} from Aggregator_DER_Bus{bus} at time {current_time}.")
    #         PVD_AGC_response = np.append(PVD_AGC_response, PVD_AGC_response_feeder)

    PVD_AGC_response_dict = {}
    #PVD_AGC_response_np = np.array([])
    while h.helicsEndpointHasMessage(epid_rAGC):
        m = h.helicsEndpointGetMessage(epid_rAGC)
        message = json.loads(m.data)
        PVD_AGC_response_feeder = message
        bus = PVD_AGC_response_feeder[0]
        PVD_AGC_response_dict[bus] = PVD_AGC_response_feeder[1:]
        print(f"Received PVD_AGC_response = {PVD_AGC_response_feeder} from Aggregator_DER_{bus} at time {current_time}.")
        tmp1 = feeder_list.index(bus)
        tmp2 = sum(num_DER_list[:tmp1])
        tmp3 = num_DER_list[tmp1]
        print("tmp1 = ",tmp1)
        print("tmp2 = ", tmp2)
        print("tmp3 = ", tmp3)
        PVD_AGC_response_np = np.array(PVD_AGC_response_feeder[1:]).astype(float)
        print(f"PVD_AGC_response_np={PVD_AGC_response_np}")
        if DER_AGC_flag:
            ss.PVD1.set(src='Pext0', idx=PVD_AGC_idx[tmp2:tmp2+tmp3], attr='v', value=PVD_AGC_response_np)
            paux0_new = ACE*AGC_participation_tgov
            ss.TurbineGov.set(src='paux0', idx=tgov_idx, attr='v', value=paux0_new)
        #PVD_AGC_response = np.append(PVD_AGC_response, PVD_AGC_response_feeder)

    # covert PVD AGC response to np
    # if len(PVD_AGC_response_dict.keys())!=0:
    #     PVD_AGC_response_list = []
    #     for bus in feeder_list:
    #         PVD_AGC_response_list = PVD_AGC_response_list + PVD_AGC_response_dict[bus]
    #
    #     PVD_AGC_response_np = np.array(PVD_AGC_response_list).astype(float)

    DER_Pmax = np.array([])
    # for bus, epid in EPID_DER_HEADROOM_R.items():
    #     while h.helicsEndpointHasMessage(epid):
    #         m = h.helicsEndpointGetMessage(epid)
    #         message = json.loads(m.data)
    #         DER_Pmax_feeder = message
    #         print(f"Received DER_Pmax = {DER_Pmax_feeder} from Aggregator_DER_Bus{bus} at time {current_time}.")
    #         DER_Pmax = np.append(DER_Pmax, DER_Pmax_feeder)

    # control the transmission thermal and DERs
    # if paux0_new is not None:
    #     ss.TurbineGov.set(src='paux0', idx=tg_idx, attr='v', value=paux0_new)

    # if DER_Pmax.shape[0] != 0:
    #     PVD_vsm_max = DER_Pmax

    mppt_current = mppt_timeseries[int(current_time), :]
    PVD_mppt_new = DER_caps * mppt_current # check PVD_mppt_new
    # PVD_mppt_new = np.minimum(DER_caps*mppt_current, PVD_vsm_max) # when there is vsm constraint

    #print(f"PVD_mppt_new = {PVD_mppt_new}")
    #print(f"DER_caps*mppt_current={DER_caps*mppt_current}")
    #print(f"PVD_vsm_max = {PVD_vsm_max}")

    # if (PVD_mppt_new <= DER_caps*mppt_current).all() and (PVD_mppt_new <= PVD_vsm_max).all():
    #     print('the minimum is correct!')
    # else:
    #     print('the minimum is wrong!')

    #print(f"PVD_mppt_new={PVD_mppt_new}")
    #ss.PVD1.set(src='pmx', idx=PVD_AGC_idx, attr='v', value=PVD_mppt_new)
    # the AGC response should stay below the Pmax: should set the andes varibles pmax for PVD1
    #print(f"PVD_AGC_response={PVD_AGC_response}")


    # if PVD_AGC_response_np.shape[0]!=0:
    #     ss.PVD1.set(src='Pext0', idx=PVD_AGC_idx[:total_num_DER], attr='v', value=PVD_AGC_response_np)
    # start the dynamic simulation
    #print(f"ss.PVD1.pmx.v={ss.PVD1.pmx.v}")
    ss.TDS.config.tf = current_time + simulation_step_time
    ss.TDS.run()
    # print(f"Bus_voltages = {ss.Bus.v.v}")
    voltage = ss.dae.ts.get_data([ss.Bus.v])[-1,:]
    print(f"current_time={current_time}")
    ACE_raw = ss.ACEc.ace.v[0]
    ACE_raw_integral = ACE_raw*simulation_step_time + ACE_raw_integral
    ACE = - KP*ACE_raw - KI*ACE_raw_integral # control settings total system
    print(f"ACE_raw = {ACE_raw}")
    print(f"ACE_raw_integral = {ACE_raw_integral}")
    #print(f"ss.ACEc.ace.v={ss.ACEc.ace.v}")
    # get DER output
    pvd1_current = ss.PVD1.Ipout_y.v
    pvd1_voltage = ss.PVD1.v.v
    pvd1_power = pvd1_current * pvd1_voltage # unit is 100 MW
    # print(f"pvd1_power={pvd1_power}")
    # if (pvd1_power <= PVD_mppt_new).all():
    #     print('The output power is correct')
    # else:
    #     print('The output power is wrong')
    bus_pvd1_power = {}

    for bus in ss.PVD1.as_df().bus.unique():
        bus_pvd1_power[str(int(bus))] = []
    for ind, bus in enumerate(ss.PVD1.as_df().bus):
        bus_pvd1_power[str(int(bus))].append(pvd1_power[ind])
    print("bus_pvd1_power",bus_pvd1_power)
    # send power to the aggregator and feeders
    # for bus, epid in EPID_DER_HEADROOM_R.items():
    #     h.helicsEndpointSendBytesTo(epid, json.dumps(bus_pvd1_power[bus]), f"Aggregator_DER_Bus{bus}_power")
    #     print(f'Sending to Aggregator_DER_Bus{bus}_power at time {current_time}: pvd1_power {bus_pvd1_power[bus]} 100 MW.')


    pvd1_power_all.append(pvd1_power) # save pvd1_power for all times
    mppt_current_all.append(mppt_current)
    #PVD_vsm_max_all.append(PVD_vsm_max)

    # every 0.5 s, set ACE signal through two endpoint
    #if current_time % ACE_interval < 1e-4: # reciving end will control this
    # send messages to Thermal controller
    # h.helicsEndpointSendBytesTo(epid_ACE, json.dumps(ACE), "Controller_Thermal_ep")
    # print(f'Sending to Controller_Thermal_ep at time {current_time}: ACE {ACE} 100 MW.')
    # send messages to DER aggregator
    for bus in ss.PVD1.as_df().bus.unique()[:num_feeder]:
        bus=str(int(bus))
        h.helicsEndpointSendBytesTo(epid_ACE, json.dumps(ACE), f"Aggregator_DER_Bus{bus}_rACE")
        print(f'Sending to Aggregator_DER_Bus{bus} at time {current_time}: ACE {ACE} 100 MW.')

    ## get the PUBLICATION OUT
    for key, pub in PUBLICATIONS.items():
        # get voltage and angle and publish
        if publications[key]['value'] == 'Voltage':
            bus_name = publications[key]['element_name']
            element_type = publications[key]['element_type']

            val1 = ss.Bus.v.v[int(bus_name)-1]
            val2 = ss.Bus.a.v[int(bus_name)-1]
            # decide when to publish based on pub_interval
            if current_time % publications[key]['pub_interval'] <= 1e-6:
                print(f"Sent voltage at time {current_time}: ({val1} pu , {val2} rad)to {element_type} {bus_name}")
                #print("Sent voltage ang at time {}: {} rad to {} {}".format(current_time, val2, element_type, bus_name))
                h.helicsPublicationPublishComplex(pub, val1, val2)

        if publications[key]['value'] == 'DER_output':
            bus_name = publications[key]['element_name']
            element_type = publications[key]['element_type']
            val1 = bus_pvd1_power[bus_name]
            if current_time % publications[key]['pub_interval'] <= 1e-6:
                print(f"Sent DER_power at time {current_time}: {val1} MW to {element_type} {bus_name}")
                h.helicsPublicationPublishVector(pub, val1)

                # if dynamic simulation can't converge, terminate the simulation
    if ss.exit_code != 0:
        raise ValueError(f"ANDES did not converge at time={current_time}")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(11, 9))
ax1.plot(ss.dae.ts.t, ss.dae.ts.get_data([ss.ACEc.f])*60)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Frequency (Hz)')

ax2.plot(ss.dae.ts.t, ss.dae.ts.get_data([ss.ACEc.ace])*100)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('ACE (MW)')

ax3.plot(ss.dae.ts.t, ss.dae.ts.get_data([ss.Bus.v],a=[0,3,8,]))
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Voltage (pu)')

ax4.plot(ss.dae.ts.t, 100*ss.dae.ts.get_data([ss.PVD1.Pext],a=[0,]))
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('PVD1 AGC signal (MW)')
#plt.show()

now = datetime.now() # record the time
append_time = now.strftime("%m_%d_%Y_%H_%M_%S")
figure_name = "cosim_results_" + append_time + ".png"
fig.savefig(os.path.join(path_to_results, figure_name), dpi=300)

# ss.TDS.load_plotter()
# ss.TDS.plt.plot(ss.PVD1.Pext, a=(0,1,2,6),ytimes=100,ylabel='AGC signal (MW)')


results_time = {}
results_time['results_time']=append_time
with open('results_time.txt','w') as outfile:
    json.dump(results_time, outfile)

#ss.TDS.load_plotter()
#print(ss.dae.ts.y.shape)
#plot the system frequency
#ss.TDS.plt.plot(ss.ACEc.f, ytimes=60, ylabel='Frequency (Hz)')

# #plot the system ACE
#ss.TDS.plt.plot(ss.ACEc.ace, ytimes=100, ylabel='ACE (MW)')
#
# #plot the bus voltage magnitude
#ss.TDS.plt.plot(ss.Bus.v, a=(0, 1, 2, 6), ylabel='Voltage (p.u.)')
# # get all voltage and save

y_object = ss.dae.ts.y

dae_ts_y = ss.dae.ts.y[:, ss.Bus.v.a[3]]
#np.savetxt("dae_ts_y.csv", dae_ts_y, delimiter=",")
# after the loop
voltage_all = ss.dae.ts.y[:, ss.Bus.v.a]

angle_all =   ss.dae.ts.y[:,ss.Bus.a.a]

voltage_PQ_3 = ss.dae.ts.y[:, ss.Bus.v.a[3]]
dae_time = ss.dae.ts.t
dae_frequency = ss.dae.ts.y[:, ss.ACEc.f.a]*60
dae_ACE = ss.dae.ts.y[:, ss.ACEc.ace.a]*100
dae_Pext_AGC = ss.dae.ts.get_data([ss.PVD1.Pext],a=[0,])*100
dae_TGOV1N_AGC = ss.dae.ts.get_data([ss.TGOV1N.paux],a=[0,])*100
dae_DER_output = ss.dae.ts.get_data([ss.PVD1.Ipout_y],a=[0,1,2,3])*ss.dae.ts.get_data([ss.PVD1.v],a=[0,1,2,3])*100

#print(f"dae_Pext_AGC.shape={dae_Pext_AGC.shape}")
allbusindex = ss.Bus.v.a

freq_ACE_np = np.hstack([dae_time.reshape(dae_time.shape[0],1),dae_frequency, dae_ACE, dae_Pext_AGC, dae_TGOV1N_AGC, dae_DER_output])

voltage_all_np = np.hstack([dae_time.reshape(dae_time.shape[0],1), voltage_all])
angle_all_np = np.hstack([dae_time.reshape(dae_time.shape[0],1), angle_all])

freq_ACE_df = pd.DataFrame(freq_ACE_np, columns=['time_s','frequency_Hz','ACE_MW', 'Pext_AGC', 'TGOV1N_AGC', 'DER_output1', 'DER_output2', 'DER_output3', 'DER_output4'])
voltage_all_df = pd.DataFrame(voltage_all_np, columns = ['time_s']+ list(allbusindex.astype(str)))
angle_all_df = pd.DataFrame(angle_all_np, columns = ['time_s']+ list(allbusindex.astype(str)))

time_list_np = np.array(time_list).reshape(len(time_list),1)

#print(f"type(mppt_current_all)")
mppt_current_all_np = np.array(mppt_current_all)
#print(f"shape={mppt_current_all_np.shape}")
# print(f"mppt_current_all_np = {mppt_current_all_np}")
# print(f"mppt_current_all_np.shape={mppt_current_all_np.shape}")
mppt_np = np.hstack([time_list_np, np.array(mppt_current_all)])

mppt_df = pd.DataFrame(mppt_np, columns = ['time_s']+ list(np.array(range(0,total_num_DER)).astype(str)))
print(f"np.array(PVD_vsm_max_all).shape= {np.array(PVD_vsm_max_all).shape}")
print(f"np.array(PVD_vsm_max_all)={np.array(PVD_vsm_max_all)}")
#print(f"time_list_np={time_list_np}")
# PVD_vsm_max_np = np.hstack([time_list_np, np.array(PVD_vsm_max_all)])
#
# PVD_vsm_max_df = pd.DataFrame(PVD_vsm_max_np, columns = ['time_s'] + list(np.array(range(0,total_num_DER)).astype(str)))


pvd1_power_np = np.hstack([time_list_np, \
                        np.array(pvd1_power_all)])

pvd1_power_df = pd.DataFrame(pvd1_power_np, columns = ['time_s']+ list(np.array(range(0,total_num_DER)).astype(str)))

dae_stack_np = np.hstack([dae_time.reshape(dae_time.shape[0],1),dae_frequency, dae_ACE, dae_Pext_AGC, dae_TGOV1N_AGC, dae_DER_output])

voltage_all_np = np.hstack([dae_time.reshape(dae_time.shape[0],1), voltage_all])

dae_stack_df = pd.DataFrame(dae_stack_np, columns=['time_s','frequency_Hz','ACE_MW', 'Pext_AGC', 'TGOV1N_AGC', 'DER_output1', 'DER_output2', 'DER_output3', 'DER_output4'])
#voltage_all_df = pd.DataFrame(voltage_all_np, columns = ['time_s']+ list(allbusindex.astype(str)))
voltage_all_df = pd.DataFrame(voltage_all_np, columns = ['time_s']+ list(ss.Bus.as_df().idx.astype(str)))
angle_all_df = pd.DataFrame(angle_all_np, columns = ['time_s']+ list(ss.Bus.as_df().idx.astype(str)))
path_to_numerical_results = os.path.join(path_to_TransmissionSim,'simulation_results')
dae_stack_df.to_csv(path_to_numerical_results + "\\dae_stack_df.csv")
voltage_all_df.to_csv(path_to_numerical_results + "\\voltage_all_df.csv")
angle_all_df.to_csv(path_to_numerical_results + "\\angle_all_df.csv")

# save two dataframe
path_to_numerical_results = os.path.join(path_to_TransmissionSim,'simulation_results')
freq_ACE_df.to_csv(path_to_numerical_results + "\\freq_ACE_df_"+append_time+".csv")
voltage_all_df.to_csv(path_to_numerical_results + "\\voltage_all_df_"+append_time+".csv")
angle_all_df.to_csv(path_to_numerical_results + "\\angle_all_df.csv")
mppt_df.to_csv(path_to_numerical_results + "\\mppt_df_"+append_time+".csv")
#PVD_vsm_max_df.to_csv(path_to_numerical_results + "\\PVD_vsm_max_df_"+append_time+".csv")
pvd1_power_df.to_csv(path_to_numerical_results + "\\pvd1_power_df_"+append_time+".csv")

received_power_array_df=pd.DataFrame(received_power_array,columns=['Received Power'])
received_power_array_df.to_csv(path_to_numerical_results + "\\received_power_df_"+append_time+".csv")

# vsm, mppt, power output

#print(y_object.shape)
# path_to_results = "simulation_results"
# np.savetxt(path_to_results + "\\voltage_all_fault.csv", voltage_all, delimiter=",")
# np.savetxt(path_to_results + "\\voltage_PQ_3_fault.csv", voltage_PQ_3, delimiter=",")
# np.savetxt(path_to_results + "\\dae_time_fault.csv", dae_time, delimiter=",")
# np.savetxt(path_to_results + "\\dae_frequency_fault.csv", dae_frequency, delimiter=",")
# np.savetxt(path_to_results + "\\dae_ACE_fault.csv", dae_ACE, delimiter=",")

# np.savetxt(path_to_results + "\\voltage_all.csv", voltage_all, delimiter=",")
# np.savetxt(path_to_results + "\\voltage_PQ_3.csv", voltage_PQ_3, delimiter=",")
# np.savetxt(path_to_results + "\\dae_time.csv", dae_time, delimiter=",")
# np.savetxt(path_to_results + "\\dae_frequency.csv", dae_frequency, delimiter=",")
# np.savetxt(path_to_results + "\\dae_ACE.csv", dae_ACE, delimiter=",")
end_time = time.time()
time_passed = round(end_time-start_time,2) # in s
time_passed_s = time_passed % 60
time_passed_m = time_passed // 60
print(f"**********************************************")
print(f"Time passed {time_passed_m} min {time_passed_s} s.")
print(f"**********************************************")

h.helicsFederateFinalize(fed)
print(f"{federate_name}: Federate finalized", flush=True)

h.helicsFederateFree(fed)
h.helicsCloseLibrary()
#print("Broker disconnected")

