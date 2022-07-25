import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DSS_solver



def AGC_prop(DER_headroom,power_demand):
    """Function to compute proportional AGC allocation"""

    DER_headroom_prop = []
    # proportional factor
    alpha = power_demand/sum(DER_headroom)

    # check if power demand is greater than available headroom limit
    if alpha > 1:
        # find the difference between demand and availbile power
        val=del_power_demand - sum(DER_headroom)
        print("DERs can't meet the request. Maximum available power could be {}. \n Difference is {} ".format(sum(DER_headroom),val) )
        print('Setting requirement to %s' %sum(DER_headroom))
        alpha=1
    DER_headroom_prop=[alpha*val for val in DER_headroom]
    return DER_headroom_prop


def AGC_limit (V_max,DER_bus_voltage,DER_sens_list):
    """compute effective headroom for each DER unit based on voltage profile"""
    DER_headroom_limit=list()
    for val in range(num_DER):
        DER_headroom_limit.append(((V_max - DER_bus_voltage[val]) / DER_sens_list[val] )/10)
    return DER_headroom_limit


def AGC_alloc (DER_headroom_limit, DER_headroom):
    """find min. of AGC_prop & AGC_limit and set that value as power increase for each unit """
    del_pmat = [min(DER_headroom_limit[i],DER_headroom[i]) for i in range(num_DER)]
    return del_pmat


def solveLPF (del_pmat, DER_bus_voltage_init):
    DER_bus_voltage_all =[]
    DER_bus_voltage=list()

    # perform M * del_pmat  
    for bus in range (num_bus):
        for col in range (num_DER):
            val = X_mat[bus,col] *  del_pmat[col]
        DER_bus_voltage_all.append(val)

    print('size : %s' %len(DER_bus_voltage_all))

    # return only DER_bus voltages
    j=0
    for idx in DER_bus_idx:
        DER_bus_voltage.append(float(DER_bus_voltage_all[idx] + DER_bus_voltage_init[j]))
        j+=1

    return DER_bus_voltage, DER_bus_voltage_all
    







## set results dir
dir_to_feeder = os.getcwd()
dir_to_results = os.path.join(dir_to_feeder,'simulation_results')


voltage0_ref=pd.read_csv('voltage_results_1_Pmin_07_06_2022_12_44_10.csv')

voltage1_d1=pd.read_csv('voltage_results_1_Pmax_07_06_2022_12_49_57.csv')
voltage1_d2=pd.read_csv('voltage_results_2_Pmax_07_06_2022_12_51_09.csv')
voltage1_d3=pd.read_csv('voltage_results_3_Pmax_07_06_2022_12_52_13.csv')
voltage1_d4=pd.read_csv('voltage_results_4_Pmax_07_06_2022_12_52_33.csv')

voltage2_d1=pd.read_csv('voltage_results_1_secant_.csv')
voltage2_d2=pd.read_csv('voltage_results_2_secant_.csv')
voltage2_d3=pd.read_csv('voltage_results_3_secant_.csv')
voltage2_d4=pd.read_csv('voltage_results_4_secant_.csv')

## default values for which results were generated
p0=500 
p1=525

# calculate deta P for Jacobian 
del_p = p1 - p0

A=[]

i=2
for idx in range(15):
    val = voltage1_d1.iloc[10,i] - voltage0_ref.iloc[10,i]
    val=val/del_p
    A.append(val)
    i+=1

B=[]

i=2
for idx in range(15):
    val = voltage1_d2.iloc[10,i] - voltage0_ref.iloc[10,i]
    val=val/del_p
    B.append(val)
    i+=1

C=[]
i=2
for idx in range(15):
    val = voltage1_d3.iloc[10,i] - voltage0_ref.iloc[10,i]
    val=val/del_p
    C.append(val)
    i+=1

D=[]
i=2
for idx in range(15):
    val = voltage1_d4.iloc[10,i] - voltage0_ref.iloc[10,i]
    val=val/del_p
    D.append(val)
    i+=1

# print(A)

# print(B)

# convert to a numpy array
X = np.array ([A,B,C,D])

# Matrix transpose
X= X.T

X_mat = np.asmatrix(X) # tangential matrix

# declare a vector to contain the change in power (deta P)
# RHS of Xmat*C = del_pmat

del_pmat = np.zeros (4)

# del_pmat[0] = 0
# del_pmat[1] = 0

# row=3
# volt_71=0
# for col in range (4):
#     volt_71=volt_71+(X_mat[row,col]*del_pmat[col])

# print(volt_71 + voltage0_ref.iloc[10,5]) 

# max_sens_idx = X_mat.argmax()
# max_sens_val = X_mat.max()

# print(max_sens_idx)
# print(max_sens_val)

# print(X_mat[11,:])

sens_der1=X_mat[3,0]
sens_der2=X_mat[9,1]
sens_der3=X_mat[11,2]
sens_der4=X_mat[14,3]

print('DER sensitivities are:\nDER1 = {} \nDER2 = {} \nDER3 = {} \nDER4 = {}'.format(sens_der1,sens_der2,sens_der3,sens_der4))

DER_sens_list=[sens_der1,sens_der2,sens_der3,sens_der4]

# DER_headroom : list containing the availble headroom reported by each DER unit

DER_headroom = [500,500,500,500]

# DER_output : current power output of each DER unit

DER_output = [500,500,500,500]

# net increase in power demand

del_power_demand,del_power_demand_const = 1050,1050

num_bus =15

# DER_headroom_prop=AGC_prop(DER_headroom,del_power_demand)

# print(sum(DER_headroom_prop))

DER_bus_voltage = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]
DER_bus_voltage_const = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]


V_max = 1.02
num_DER = 4

DER_bus_idx = [3,9,11,14]

# DER_headroom_limit = AGC_limit(V_max,DER_bus_voltage,DER_sens_list)

# print('DER headroom limit is %s\n %s\n %s\n %s'%(DER_headroom_limit[0],DER_headroom_limit[1],DER_headroom_limit[2],DER_headroom_limit[3]))

# del_pmat = AGC_alloc (DER_headroom_limit, DER_headroom_prop)

# print('Initial power allocation is %s' %del_pmat)

# print('Sum of initial power allocation is %s' %sum(del_pmat))

initial_output= sum(DER_output)

print('Initial total power ooutput of DERs is %s' %initial_output)

print('~~~~ Entering Main Loop ~~~~~~')

# Flag to represent AGC allocation status
# becomes False when AGC request is met or system limmit is reached 
AGC_undelivered = True

index =0

while AGC_undelivered:

    # find the proportional headroom for each DER unit
    DER_headroom_prop=AGC_prop(DER_headroom,del_power_demand)

    print(sum(DER_headroom_prop))

    print(DER_headroom_prop)

    # DER_bus_voltage = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]

    # find effective headroom for each DER unit
    DER_headroom_limit = AGC_limit(V_max,DER_bus_voltage,DER_sens_list)
    print('DER headroom limit is %s\n %s\n %s\n %s'%(DER_headroom_limit[0],DER_headroom_limit[1],DER_headroom_limit[2],DER_headroom_limit[3]))

    # Allocate AGC to DER units based on AGC_prop & AGC_limit
    DER_headroom = AGC_alloc (DER_headroom_limit, DER_headroom)

    print('DER headroom set as: %s' %DER_headroom)   

    # print('Sum of initial power allocation is %s' %sum(del_pmat))

    DER_sat_list = []
    P_diff=0
    # allocate appropriate ower to each unit
    for i in range (num_DER):
        # check if eff < prop
        if DER_headroom[i] <= DER_headroom_prop[i]:

            print(DER_headroom_limit[i])

            print(DER_headroom_prop[i])
            # find power difference
            P_diff+=(DER_headroom_prop[i]-DER_headroom[i])
            # DER_sat_list.append(i)
            # increase the DER power output by DER_limit
            DER_output[i] = DER_output[i] + DER_headroom[i]
            # set DER headroom to 0!
            DER_headroom[i] = 0
            # set del_pmat as DER_limit
            del_pmat[i] = DER_headroom[i]
            # num_DER -=1
            # DER_headroom.pop[i]
        else:
            # increase DER power by DER_prop
            DER_output[i] = DER_output[i] + DER_headroom_prop[i]
            # set the available headroom as the diff. between available headroom & DER_prop 
            DER_headroom[i] = DER_headroom[i] - DER_headroom_prop[i]
            # set del_pmat as DER_prop
            del_pmat[i] = DER_headroom_prop[i]  
    
    print('~~~~DER output %s' %DER_output)

    print('DER headroom updated as: %s' %DER_headroom)
    del_power_demand = P_diff # update the power demand as the total undelivered power

    # perform linear power flow for node voltage 

    DER_bus_voltage, DER_bus_voltage_all = solveLPF(del_pmat, DER_bus_voltage)

    # DER_bus_voltage = [DER_bus_voltage_const[i]+DER_bus_voltage[i] for i in range (num_DER)]

    print('bus voltages are %s' %DER_bus_voltage)

    if (P_diff == del_power_demand_const) :
        print("DER max. limit reached. \n Total power delivered is {}".format(sum(DER_output) - initial_output))
        AGC_undelivered = False
    elif P_diff == 0:
        AGC_undelivered = False
    else:
        continue
        
print('DER output %s' %DER_output)

print('DER_headroom %s' %DER_headroom)

DER_bus_voltage_DSS_all = DSS_solver.solveDSS(DER_output)



DER_bus_voltage_DSS = [DER_bus_voltage_DSS_all[idx+2] for idx in DER_bus_idx]

# calculate error

bus_voltage_error_max =

bus_vltage_error_avg  =  














