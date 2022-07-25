import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from CPDS_validation_cosim.Distribution.Bus11.Bus11 import DER_output

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

# DER_sens_list.sort()

DER_headroom = [500,500,500,500]

DER_output = [500,500,500,500]

del_power_demand = 1050

# del_pmat [0] = 

# sens_min=min(DER_sens_list)
# DER_sens_list_norm= [val/sens_min for val in DER_sens_list]
# gamma = [1/val for val in DER_sens_list_norm]
# DER_headroom_effect =[]
# j=0
# for val in DER_headroom:
#     DER_headroom_effect.append(gamma[j]*val)
#     j+=1
# del_p_AGC = 100
# alpha = del_p_AGC / (sum(DER_headroom_effect))
# del_pmat = [val*alpha for val in DER_headroom_effect]
# print(sum(del_pmat))

## equal headroom division
alpha = del_power_demand / sum(DER_headroom)

if alpha > 1:
    val=del_power_demand - sum(DER_headroom)
    print("DERs can't meet the request. Maximum available power could be {}. \n Difference is {} ".format(sum(DER_headroom),val) )
    print('Setting requirement to %s' %sum(DER_headroom))
    alpha=1



DER_headroom_prop = [alpha*val for val in DER_headroom]

# print(DER_headroom_prop)

print(sum(DER_headroom_prop))
    

DER_headroom_limit = []

DER_bus_voltage = [voltage0_ref.iloc[10,5],voltage0_ref.iloc[10,11],voltage0_ref.iloc[10,13],voltage0_ref.iloc[10,16]]

V_max = 1.02
num_DER = 4

# i=0
for val in range(num_DER):
    DER_headroom_limit.append(((V_max - DER_bus_voltage[val]) / DER_sens_list[val] )/10)
    # i+=1

print('DER headroom limit is %s\n %s\n %s\n %s'%(DER_headroom_limit[0],DER_headroom_limit[1],DER_headroom_limit[2],DER_headroom_limit[3]))

# print(DER_bus_voltage)
## del_pmat is the right hand matrix of equation [del_v] = M * del_pmat 

del_pmat = [min(DER_headroom_limit[i],DER_headroom_prop[i]) for i in range(num_DER)]
# print('~~~del_pmat is %s '%del_pmat)

print('Initial Power allocation is %s' %del_pmat)
print('Initial P_inj = %s' %sum(del_pmat))


val =0
DER_sat=[]
DER_sat_val=[]
DER_sat_range=[]
for i in range (num_DER):
    if DER_headroom_limit[i] < DER_headroom_prop[i]:
        DER_sat.append(i)
        DER_sat_val.append(DER_headroom_prop[i] - DER_headroom_limit[i]) 
        val = DER_headroom_prop[i] - DER_headroom_limit[i]
    DER_sat_range.append(DER_headroom_limit[i] - DER_headroom_prop[i])




for i in range (4):
    if  i not in DER_sat:
        del_pmat[i] = del_pmat[i] + abs(val/3)
    else:
        continue


print(del_pmat)

DER_output_new = [del_pmat[i]+DER_output[i] for i in range(num_DER)]

print('NEW DER output is %s' %DER_output_new)


# del_pmat = [DER_output_new[i]-DER_output[i] for i in range (num_DER)]
# print(del_pmat)

## Calculate bus voltages at new values of DER_output using linearized model

voltage_sim = []
val = 0
j=2
for row in range (15):
    for col in range (num_DER):
        val = val + X_mat[row,col]*del_pmat[col]
        # print(val)
    voltage_sim.append(val + voltage0_ref.iloc[10,j])
    val = 0
    j+=1

print(voltage_sim)

voltage_results = pd.DataFrame(voltage_sim)
# dir_to_results = os.path.join(dir_to_feeder, "simulation_results")
# voltage_results.to_csv(dir_to_results+'\\voltage_results_'+str(active_der)+'_'+active_power+'_'+append_time+'.csv')
voltage_results.to_csv(dir_to_results+'\\jacobian_results_'+str(del_power_demand)+'.csv')


# ## plot results
# r1=pd.read_csv('voltage_results_525.csv')
# r2=pd.read_csv('voltage_results_550.csv')
# r3=pd.read_csv('voltage_results_600.csv')
# r4=pd.read_csv('voltage_results_725.csv')
# r5=pd.read_csv('voltage_results_1050.csv')

# # print(r1.iloc(3,3))

# plt.figure (1)

# # for i in range (15)
# j=3
# # x=[632,670,671,680,633,645,646,692,675,684,611,652,634,650]
# x=[100,200,400,900,1050]
# # y=[r1.iloc(4,3),r1.iloc(4,3),r1.iloc(4,3)]

# for xx in range (len(x)):
#     plt.plot(x[xx],r1.iloc[4,j])
#     j+=1


# plt.show()
