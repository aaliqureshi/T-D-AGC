from datetime import datetime
import json
import os
import sys
import matplotlib
import numpy as np
import opendssdirect as dss
from opendssdirect.utils import run_command
import pandas as pd

import andes
import matplotlib.pyplot as plt
from sympy import degree

dir_to_feeder = os.getcwd()
print(f"python_dir={os.getcwd()}")
dir_to_project = os.path.join(dir_to_feeder,'..','..',)
dir_to_Algorithm = os.path.join(dir_to_project, 'Module_files')
sys.path.append(dir_to_Algorithm)

path_to_tx_xlsx=os.path.join(dir_to_feeder,'..','..','Network_models','Transmission')


V_low=0.95
V_high=1.05

################# read in config file ###########################################
print("reading json")
filename = "Distribution_setup.json"
with open(filename) as f:
    data = json.loads(f.read())
    federate_name = data["name"]
    total_time = data["total_time"]
    num_DER = data["num_DER"]
    DER_output_MW_prescale = data["DER_output_MW_prescale"]
    DER_rating_MW_prescale = data["DER_rating_MW_prescale"]
    simulation_step_time = data["simulation_step_time"]
    subscriptions = data["subscriptions"]
    publications = data["publications"]
    scale_P = data["scale_P"]
    scale_Q = data["scale_Q"]
    v0 = data["v0"]
    a0 = data["a0"]
print("reading json okay!")


############ setting andes ####################
tx_xlsx= 'ieee14_pvd1_1DistBu1.xlsx'
tx_path=os.path.join(path_to_tx_xlsx,tx_xlsx)
ss=andes.run(tx_path,setup=False)
ss.setup()

########################################


consistent_random_object = np.random.RandomState(321)
load_profile_random = consistent_random_object.normal(1, 0.0, int(total_time))

## OpenDSS setting up the case
feeder_name = federate_name
print('------ Running the {} feeder in opendss'.format(feeder_name))
run_command('compile DSSfiles/Master.dss')
if dss.Text.Result() == '':
    print('------ Success for the test run -----')
else:
    print('------ Opendss failed ----')
    print(f'Error is "{dss.Text.Result()}"!')
print(f"After opendss run os.getcwd()={os.getcwd()}")

# Batch edit objects in the same class (i.e. for each load, set Vminpu=0.9)
# Vminpu: Minimum per unit voltage for which the MODEL is assumed to apply. Below this value, the load model reverts to a constant impedance model.
run_command("BatchEdit Load..* Vminpu=0.9")

#  A LoadShape object consists of a series of multipliers, typically ranging from 0.0 to 1.0 that are applied to the base kW values of the load to represent variation of the load over some time period.
run_command("New Loadshape.dummy npts=60 sinterval=1 Pmult=[file=dummy_profile.txt]")
run_command("BatchEdit Load..* Yearly=dummy")

#  Do a solution following the yearly load curves. The solution is repeated as many times as the specified by the Number= option. Each load then follows its yearly load curve
# run_command("set mode=yearly number=60 stepsize=1s")
run_command("set mode=snapshot")


dss.Solution.ControlMode(2)

# Executes the solution mode specified by the Set Mode = command. It may execute a single solution or hundreds of solutions
print(f"dss.Solution.Mode()={dss.Solution.Mode()}")


print(f"dss.Solution.ControlMode()={dss.Solution.ControlMode()}")


# get load kw values
BASEKV = dss.Vsources.BasekV()
loadname_all=[]
loadkw_all=[]
loadkvar_all=[]
num_loads = dss.Loads.Count()
dss.Loads.First() # Set first Load active; returns 0 if none.
for i in range(num_loads):
    loadname = dss.Loads.Name() # Get/set the name of the active Load
    #print(f'loadname={loadname}')
    loadkw = dss.Loads.kW() # Set kW for active Load. Updates kvar based on present PF.
    loadkvar = dss.Loads.kvar() # Get/set kvar for active Load. If set, updates PF based on present kW.
    loadname_all.append(loadname)
    loadkw_all.append(loadkw)
    loadkvar_all.append(loadkvar)
    dss.Loads.Next() # Sets next Load active; returns 0 if no more


loadkw_dict = dict(zip(loadname_all, loadkw_all))
loadkvar_dict = dict(zip(loadname_all, loadkvar_all))
allnodenames = dss.Circuit.AllNodeNames()
num_nodes = len(allnodenames)

load_mult = 1.0111

dss.Loads.First()
# sum = 0

for j in range(num_loads):
    loadname = dss.Loads.Name()
    # print(f"{loadname} kw={dss.Loads.kW()}")
    # text = run_command("? Load.p_perturb.enabled")

    # now calculate kw and kvar based on the csv
    #load_mult = load_profile_random[int(current_time)]
    loadkw_new = load_mult * loadkw_dict[loadname]
    loadkvar_new = load_mult * loadkvar_dict[loadname]
    # now change opendss load object
    run_command('edit load.{ln} kw={kw}'.format(ln=loadname, kw=loadkw_new))
    run_command('edit load.{ln} kvar={kw}'.format(ln=loadname, kw=loadkvar_new))
    # sum+=loadkw_new
    dss.Loads.Next()


print(f"Total power base line is {dss.Circuit.TotalPower()}") # (read-only) Total power, watts delivered to the circuit

P0, Q0 = dss.Circuit.TotalPower() 

## setup ANDES power flow to get v0 and a0

ss.PQ.alter(src='p0',idx='PQ_8',value=-P0/1e5)
ss.PQ.alter (src='q0',idx='PQ_8',value=-Q0/1e5)
ss.PFlow.run()
v0=ss.Bus.v.v[7]
a0=np.degrees(ss.Bus.a.v[7])
# angle_deg=np.degrees(a_array[7])
# voltage=v_array[7]

## end ANDES power flow

current_time = 0
allbusmagpu_ts = []
totalpower_ts = []

consistent_random_object = np.random.RandomState(321)
index_DERs = consistent_random_object.randint(len(allnodenames), size=num_DER) # return random integers from low (inclusive) to high (exclusive)


DER_output = np.ones(num_DER)*(DER_output_MW_prescale*1e3/num_DER)/scale_P

DER_output_limit=np.ones(num_DER)*(DER_rating_MW_prescale*1e3/num_DER)/scale_P

# DER_output=np.zeros(num_DER)

print(f"DER output after scale in kw each = {DER_output[0]} of {num_DER} DERs.")

dss.Vsources.PU(v0) # Per-unit value of source voltage
dss.Vsources.AngleDeg(np.degrees(a0)) # Phase angle of first phase in degrees

i=1
for ind in index_DERs:
    nodename = allnodenames[ind]
    # print(nodename)
    dss.Circuit.SetActiveBus(nodename)
    buskv = dss.Bus.kVBase()
    run_command(f'New generator.der_{i} bus1={nodename} Phases=1 Conn=Wye Model=1 kV={buskv} kw={DER_output[i-1]} kvar=0')
    i=i+1
run_command("BatchEdit Generator..* Yearly=dummy")

DER_output_pre = None
voltage = v0

print(dss.Circuit.TotalPower())



print(f"index_DERs = {index_DERs}")

# for der_inc_percent in range(10):
der_output_array=[]

# for request_time in np.arange(0, 1):

request_MW=1500

# request_time = 2

# power1,power2,power3,power4=[],[],[],[]
# voltage1,voltage2,voltage3,voltage4=[],[],[],[]

total_time=30

# DER_output[0]=DER_output_limit[0]

active_der=1

power0=np.array([])
power1=np.array([])
voltage0=np.array([])
voltage1=np.array([])

for gen in range (num_DER):

    for request_time in np.arange(0, total_time, simulation_step_time):
        # while current_time < request_time:
        #     current_time = h.helicsFederateRequestTime(fed, request_time)
        #     #print(current_time)
        print("==============================")
        print(f"current_time={request_time}")
        # add DER power
        # while h.helicsEndpointHasMessage(epid):
        #     m = h.helicsEndpointGetMessage(epid)
        #     message = json.loads(m.data)
        #     DER_output = np.array(message)
        #     print(f"Received message at time {current_time}: DER_power {DER_output} in kw")
        # based on the DER_output modify DER

        if DER_output_pre is not None:# 
            if np.isscalar(DER_output_pre):
                DER_output = (DER_output_pre/num_DER)*np.ones(num_DER)
            else:
                # DER_output = (sum(DER_output_pre)/num_DER)*np.ones(num_DER)
                # DER_output = request_MW - sum(DER)
                DER_output = DER_output_pre
            # print(f'scale_P={scale_P}')
        print(f"DER_output={DER_output} at time {request_time}")

        der_output_array.append(DER_output[0])

        j=1
        for ind in index_DERs:
            nodename = allnodenames[ind]
            run_command(f'Edit generator.der_{j} kw={DER_output[j-1]/3}') # opendss positive sequence should defined as 1 phase
            j=j+1

        dss.run_command("Solve number=1 stepsize=1s")
        # reply = run_command("? Load.D862_838sb.Vminpu")
        # print(f"reply={reply}")
        # reply = run_command("? Regcontrol.creg1a.TapNum")
        # print(f"TapNum reply={reply}")
        #run_command("show taps")
        # save some results
        allbusmagpu = dss.Circuit.AllBusMagPu()
        allbusmagpu_ts.append(allbusmagpu)
        totalpower_ts.append(dss.Circuit.TotalPower())

        # total power adjust amount of total DER_output
        S = dss.Circuit.TotalPower() # the total power is negative
        S_net=dss.Circuit.TotalPower()

        S[0] = S[0] - np.sum(DER_output) # adjust
        #
        print(f"======Total power demand={dss.Circuit.TotalPower()}")
        # print(f"DER power is {np.sum(DER_output)}")
        # print(f"Overall after adjustment S={S}")
        P = -S[0]
        Q = -S[1]

        P = P*scale_P
        Q = Q*scale_Q

        print('Net active power at time step %s is: %s' %(current_time,P))
        # print('Net reactive power comsumption at time step %s is: %s' %(current_time,Q))

        #Xeq = Q/voltage**2
        # now publish the results
        # h.helicsPublicationPublishComplex(pub, P, Q) # Xeq was Q
        # print("Sent Active power at time {}: {} kw".format(current_time, P))
        # print("Sent Reactive power at time {}: {} kvar".format(current_time, Q))

        allbusmagpu_base = np.array(allbusmagpu)
        # print(f'min of voltage = {min(allbusmagpu_base)}')
        # print(f'max of voltage = {max(allbusmagpu_base)}')

        allbusmagpu_base = np.array(allbusmagpu)
        V_min=np.amin(allbusmagpu_base)
        V_max=np.amax(allbusmagpu_base)
        V_min_idx=np.argmin(allbusmagpu_base)
        V_max_idx=np.argmax(allbusmagpu_base)

        V_min=np.amin(allbusmagpu_base)
        V_min_idx=np.argmin(allbusmagpu_base)
        V_max_idx=np.argmax(allbusmagpu_base)

        # print('Bus %s has the lowest voltage of %s'%(V_min,allnodenames[V_min_idx]))
        # print('Bus %s has the highest voltage of %s'%(V_max,allnodenames[V_max_idx]))

        for zz in range(len(allbusmagpu_base)):
            if allbusmagpu_base[zz] < V_low:
                print('Bus %s Voltage is lower than nominal'%allnodenames[zz])
            elif allbusmagpu_base[zz] > V_high:
                print("Bus %s Voltage is higher than nominal value" %allnodenames[zz])

        
        # if request_time==0:
        #     angle_deg=np.degrees(a0)
        #     voltage=v0
        # else:
        ss.PQ.alter(src='p0',idx='PQ_8',value=-S_net[0]/1e5)
        ss.PQ.alter(src='q0',idx='PQ_8',value=-S_net[1]/1e5)
            # pq_andes=ss.PQ.get('p0','PQ_8',attr='vin')
            # ss.setup()
        ss.PFlow.run()
            # print('p0 is {}'.format(ss.PQ.get(src='p0',idx='PQ_8',attr='vin')))
            # print('Ppf is {}'.format(ss.PQ.get(src='Ppf', idx='PQ_8', attr='v')))
        voltage=ss.Bus.v.v[7]
        angle_deg=np.degrees(ss.Bus.a.v[7])
            # angle_deg=np.degrees(a_array[7])
            # voltage=v_array[7]
        
        dss.Vsources.AngleDeg(angle_deg)
        dss.Vsources.PU(voltage)

        if current_time == 10:
            np.append(voltage0,allbusmagpu)


    # if request_time%2 ==0 and request_time!=0:
    #     DER_output_pre[0]=DER_output_limit[0]
    # elif request_time==0:
    #     DER_output_pre=DER_output

    


    
    # ss.PQ.alter('p0','PQ_8',)

    ## add andes power flow model ##
    

    # if current_time % 10 < 1e-4:
    #     # calculate vsm
    #     allbusmagpu_base = np.array(allbusmagpu)
    #     V_min=np.amin(allbusmagpu_base)
    #     V_max=np.amax(allbusmagpu_base)
    #     V_min_idx=np.argmin(allbusmagpu_base)
    #     V_max_idx=np.argmax(allbusmagpu_base)
    #     print('Bus %s has the lowest voltage of %s'%(V_min,allnodenames[V_min_idx]))
    #     print('Bus %s has the highest voltage of %s'%(V_max,allnodenames[V_max_idx]))

    #     for zz in range(len(allbusmagpu_base)):
    #         if allbusmagpu_base[zz] < V_low:
    #             print('Bus %s Voltage is lower than nominal'%allnodenames[zz])
    #         elif allbusmagpu_base[zz] > V_high:
    #             print("Bus %s Voltage is higher than nominal value" %allnodenames[zz])



    #     print("********* getting vsm ****************")
    #     #print(f'before control mode = {dss.Solution.ControlMode()}')
    #     print(f'min of voltage = {min(allbusmagpu_base)}')
    #     print(f'max of voltage = {max(allbusmagpu_base)}')
    #     vsm = vsm_utils.get_vsm(dss, feeder_name, index_DERs, allbusmagpu_base)
    #     #print(f'after control mode = {dss.Solution.ControlMode()}')
    #     #print(f'after dss.Loads.Count()={dss.Loads.Count()}')
    #     print("***************************************")

        # send vsm to aggregator
        # to_endpoint = "Aggregator_DER_" + feeder_name + "_feeder"
        # h.helicsEndpointSendBytesTo(epid, json.dumps([vsm.tolist(), allbusmagpu_base.tolist()]), to_endpoint)
        # print(f'Sending vsm to Aggregator_DER for {feeder_name} at time {current_time}: showing a block of vsm {vsm[index_DERs,:]}')

    # send this vsm and v_base to aggregator

    # now get sub the results
    # for key, sub in SUBSCRIPTIONS.items():
    #     if subscriptions[key]['value'] == 'Voltage':
    #         val = h.helicsInputGetComplex(sub)
    #         print("Received voltage mag at time {}: {} pu".format(current_time, val[0]))
    #         print("Received voltage ang at time {}: {} rad".format(current_time, val[1]))
    #         voltage, angle_rad = val
    #         # convert angle to degree
    #         angle_deg = np.degrees(angle_rad)
    #         # change the substation voltage and angle based on sub
    #         # print(f'current time={current_time}')
    #         if current_time == 0:
    #             print('manually set up first step voltage')
    #             angle_deg = np.degrees(a0)
    #             voltage = v0
    #         dss.Vsources.AngleDeg(angle_deg)
    #         dss.Vsources.PU(voltage)
    #     # change load according to load profile
    #     if subscriptions[key]['value'] == 'DER_output':
    #         val = h.helicsInputGetVector(sub)
    #         print("Received DER_output at time {}: {} MW".format(current_time, val))
    #         if current_time == 0:
    #             print('Not change DER_output at time 0')
    #         else:
    #             DER_output_pre = np.array(val)
    # change load according to load profile
    # dss.Loads.First()
    # for j in range(num_loads):
    #     loadname = dss.Loads.Name()
    #     #print(f"{loadname} kw={dss.Loads.kW()}")
    #     #text = run_command("? Load.p_perturb.enabled")
    #
    #     # now calculate kw and kvar based on the csv
    #     load_mult = load_profile_random[int(current_time)]
    #     loadkw_new = load_mult * loadkw_dict[loadname]
    #     loadkvar_new = load_mult * loadkvar_dict[loadname]
    #     # now change opendss load object
    #     run_command('edit load.{ln} kw={kw}'.format(ln=loadname, kw=loadkw_new))
    #     run_command('edit load.{ln} kvar={kw}'.format(ln=loadname, kw=loadkvar_new))
    #     dss.Loads.Next()

now = datetime.now() # record the time
append_time = now.strftime("%m_%d_%Y_%H_%M_%S")

voltage_results = pd.DataFrame(allbusmagpu_ts, columns=allnodenames)
dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
voltage_results.to_csv(dir_to_results+'\\voltage_results_'+''+append_time+'.csv')

results_time = {}
results_time['results_time']=append_time
with open(os.path.join(dir_to_results,'results_time.txt'), 'w') as outfile:
    json.dump(results_time, outfile)

# h.helicsFederateFinalize(fed)

# h.helicsFederateFree(fed)
# h.helicsCloseLibrary()
print("Federate finalized")

# kk=voltage_results[allnodenames[0]]

# print(voltage_results.columns[4])

plt.figure(1)
plt.plot(voltage_results.iloc[:,4][2:])
plt.legend(allnodenames[6])
plt.figure(2)
plt.plot(der_output_array[2:])


plt.figure(3)

y_ax=np.array(voltage_results.iloc[:,5][2:])
plt.plot(der_output_array[2:],y_ax)

plt.figure(4)
plt.plot(voltage_results[:][2:])
# # voltage_results[allnodenames[1]].plot()
plt.show()



