import numpy as np
import DSS_PF
import os
import pandas as pd


def calc (del_agc,type,DER_idx,var_dict,num_nodes,P_diff):
    dir_to_feeder = os.getcwd()
    # print(dir_to_feeder)
    dir_to_results = os.path.join(dir_to_feeder,'..','simulation_results')
    print(dir_to_results)

    voltage0_ref=pd.read_csv(dir_to_results+'\\initial_voltage_'+str(del_agc)+'.csv')

    for idx in range (len(DER_idx)):
        var_dict[idx] = pd.read_csv(dir_to_results+'\\'+type+'_results_'+str(del_agc)+'_'+str(idx)+'.csv')


    jac_cols={}

    for row in range(len(DER_idx)):
        arr=list()
        for col in range (num_nodes):
            val = (var_dict[row].iloc[col,1] - voltage0_ref.iloc[4,col+1])/P_diff[row]
            arr.append(val)
        jac_cols[row]=arr

    X = np.empty([len(DER_idx),len(jac_cols[0])])
    
    j=0
    for idx in range(len(DER_idx)):
        X[j]=jac_cols[idx]
        j+=1

    X.size
    X.shape
    X=X.T

    X_mat = np.asmatrix(X)

    j=0
    X_sens=list()
    for bus in DER_idx:
        X_sens.append(X[bus,j])
        j+=1

    return X_mat,X_sens

    



def Tan(v0,a0,DER_idx,DER_pert,DER_output,del_agc):
    type='tangent'
    var_dict ={}
    DER_out = DER_output[:]
    P_diff = [DER_pert[i] - DER_output[i] for i in range(len(DER_idx))]
    for i in range (len(DER_idx)):
        DER_out[i]=DER_pert[i]
        print(DER_out)
        num_nodes,_=DSS_PF.solvePF (DER_out,DER_idx,del_agc,type,i,store=1)
        DER_out[i]=DER_output[i]
        val='vt_d' + str(i)
        var_dict[i] = val
    T_mat,T_sens=calc(del_agc,type,DER_idx,var_dict,num_nodes,P_diff)
    return T_mat, T_sens



def Sec(v0,a0,DER_idx,DER_max,DER_output,del_agc):
    type='secant'
    var_dict={}
    DER_out = DER_output[:]
    P_diff=[DER_max[i]-DER_output[i] for i in range(len(DER_output))]
    for i in range (len(DER_idx)):
        DER_out[i]=DER_max[i]
        print(DER_out)
        num_nodes,_=DSS_PF.solvePF (DER_out,DER_idx,del_agc,type,i,store=1)
        DER_out[i]=DER_output[i]
        val='vs_d'+str(i)
        var_dict[i]= val
    S_mat,S_sens=calc(del_agc,type,DER_idx,var_dict,num_nodes,P_diff)

    return S_mat,S_sens

