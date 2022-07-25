import os
import pandas as pf
import numpy as np
import matplotlib.pyplot as plt



def AGC_prop(DER_headroom,power_demand):
    DER_headroom_prop = []
    alpha = power_demand/sum(DER_headroom)
    if alpha > 1:
        val=del_power_demand - sum(DER_headroom)
        print("DERs can't meet the request. Maximum available power could be {}. \n Difference is {} ".format(sum(DER_headroom),val) )
        print('Setting requirement to %s' %sum(DER_headroom))
        alpha=1
    DER_headroom_prop=[alpha*val for val in DER_headroom]
    return DER_headroom_prop


def AGC_limit (V_max,DER_bus_voltage,DER_sens_list):
    DER_headroom_limit=list()
    for val in range(num_DER):
        DER_headroom_limit.append(((V_max - DER_bus_voltage[val]) / DER_sens_list[val] )/10)
    return DER_headroom_limit


def AGC_alloc (DER_headroom_limit, DER_headroom_prop):
    del_pmat = [min(DER_headroom_limit[i],DER_headroom_prop[i]) for i in range(num_DER)]
    

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

p0=500 ## default values for which results were generated
p1=525

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
X = np.array ([A,B,C,D])
X= X.T
# print(X)

X_mat = np.asmatrix(X) # tangential matrix













# print(X_mat)

del_pmat = np.zeros ((4,1))

del_pmat[0] = 0
del_pmat[1] = 0

row=3
volt_71=0
for col in range (4):
    volt_71=volt_71+(X_mat[row,col]*del_pmat[col])

print(volt_71 + voltage0_ref.iloc[10,5]) 

max_sens_idx = X_mat.argmax()
max_sens_val = X_mat.max()

# print(max_sens_idx)
# print(max_sens_val)

# print(X_mat[11,:])

sens_der1=X_mat[3,0]
sens_der2=X_mat[9,1]
sens_der3=X_mat[11,2]
sens_der4=X_mat[14,3]

print('DER sensitivities are:\nDER1 = {} \nDER2 = {} \nDER3 = {} \nDER4 = {}'.format(sens_der1,sens_der2,sens_der3,sens_der4))

DER_sens_list=[sens_der1,sens_der2,sens_der3,sens_der4]


DER_headroom = [500,500,500,500]

DER_output = [500,500,500,500]

del_power_demand = 1050

DER_headroom_prop=AGC_prop(DER_headroom,del_power_demand)

print(sum(DER_headroom_prop))

DER_bus_voltage = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]

V_max = 1.02
num_DER = 4

DER_headroom_limit = AGC_limit(V_max,DER_bus_voltage,DER_sens_list)

print('DER headroom limit is %s\n %s\n %s\n %s'%(DER_headroom_limit[0],DER_headroom_limit[1],DER_headroom_limit[2],DER_headroom_limit[3]))

del_pmat = AGC_alloc (DER_headroom_limit, DER_headroom_prop)
