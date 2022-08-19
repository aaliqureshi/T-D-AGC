from tkinter import font
import pandas as pd
import csv
import matplotlib.pyplot as plt

def plotty (error,DER_output_initial,DER_output_final,key):
# def plotty(error):
    nodes=pd.read_csv(r"C:\Users\aaliq\Documents\AGC Codes\T-D-AGC\CPDS_validation_cosim\Distribution\Bus11\simulation_results\nodes_1050.csv")
    j=0
    a_idx,b_idx,c_idx =list(),list(),[]
    x_axis_a,x_axis_b,x_axis_c=[],[],[]
    # a,b,c=0,0,0
    for idx in range(len(nodes)):
        val = nodes.iloc[j,1]
        val = str(val)
        node,phase=val.split('.')
        if int(phase) == 1:
            a_idx.append(j)
            x_axis_a.append(val)
        elif int(phase) == 2:
            b_idx.append(j)
            x_axis_b.append(val)
        else:
            c_idx.append(j)
            x_axis_c.append(val)
        j+=1

# def plotty (error):
    # x_axis1=['src.1','800.1','806','808','810','812','814','814r','850','816','818','824','820','822','826','828','830','854','832','858','834','860','842','836','840','862','844','846','848','852r','888','856','852','864','838','890']
    # x_axis2=['src.2','800.2','802','806','808','810','812','814','814r','850','816','818','824','820','822','826','828','830','854','832','858','834','860','842','836','840','862','844','846','848','852r','888','856','852','864','838','890']
    # x_axis3=['src.3','800.3','802','806','808','810','812','814','814r','850','816','818','824','820','822','826','828','830','854','832','858','834','860','842','836','840','862','844','846','848','852r','888','856','852','864','838','890']
    # a,b,c=0,1,2
    # error_a,error_b,error_c=list(),list(),list()
    # while (a<len(error)-1):
    #     error_a.append(error[a])
    #     a+=3
    # while (b<len(error)-1):
    #     error_b.append(error[b])
    #     b+=3
    # while (c<len(error)-1):
    #     error_c.append(error[c])
    #     c+=3
    error_a = [error[idx] for idx in a_idx]
    error_b = [error[idx] for idx in b_idx]
    error_c = [error[idx] for idx in c_idx]
    power_injected = abs(sum(DER_output_final)-sum(DER_output_initial))

    # if key == 'TP':
    #     title = 'Tangential'
    plt.figure(1)
    plt.plot(error_a,"--",linewidth =3)
    plt.xticks(ticks = range(len(x_axis_a)),labels=x_axis_a,rotation='vertical')
    # plt.legend(('%s error (tangent line approx.'))
    plt.xlabel("Bus Number",fontweight='bold')
    plt.ylabel("Voltage error [%] at phase A",fontweight='bold')
    plt.title("%s approx. DER dispatched power: %s."%(key,power_injected),fontweight='bold')
    pic_title=key+'_A'
    plt.savefig(pic_title,dpi=300)
    plt.clf()
    # plt.show()

    plt.figure(2)
    plt.plot(error_b,"--",linewidth =3)
    plt.xticks(ticks = range(len(x_axis_b)),labels=x_axis_b,rotation='vertical')
    # plt.legend(('%s error (tangent line approx.'))
    plt.xlabel("Bus Number",fontweight='bold')
    plt.ylabel("Voltage error [%] at phase B",fontweight='bold')
    plt.title("%s approximation. DER dispatched power is %s."%(key,power_injected),fontweight='bold')
    pic_title=key+'_B'
    plt.savefig(pic_title,dpi=300)
    plt.clf()
    # plt.show()
    

    plt.figure(3)
    plt.plot(error_c,"--",linewidth =3)
    plt.xticks(ticks = range(len(x_axis_c)),labels=x_axis_c,rotation='vertical')
    # plt.legend(('%s error (tangent line approx.'))
    plt.xlabel("Bus Number",fontweight='bold')
    plt.ylabel("Voltage error [%] at phase C",fontweight='bold')
    plt.title("%s approximation. DER dispatched power is %s."%(key,power_injected),fontweight='bold')
    pic_title=key+'_C'
    plt.savefig(pic_title,dpi=300)
    plt.clf()
    # plt.show()



# error = [1,2,3,4,5,6,7,8,9,10,11,12,13]
# plotty(error)
def plot_ratio (key,method):
    power_demand=list()
    avg_error=list()
    max_error=list()
    file1 = key+'_errors.csv'
    with open(file1,newline='') as csvfile:
        rr=csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in rr:
            power_demand.append(row[0])
            avg_error.append(row[1])
            max_error.append(row[2])
    
    plotter(power_demand,avg_error,'Average',key,method)
    plotter(power_demand,max_error,'Maximum',key,method)

def plotter(power_demand,error,typee,key,method):
    plt.figure()
    plt.stem(error)
    plt.xticks(ticks = range(len(power_demand)), labels=power_demand)
    plt.xlabel('Power Demand (kW)', fontweight='bold')
    plt.ylabel(f'{typee} Voltage Error (%) pu ',fontweight='bold')
    plt.title(f'{key} approximation',fontweight='bold')
    pic_title=key+'_'+typee+'_'+method
    plt.savefig(pic_title,dpi=300)
    plt.clf()
    # plt.show()





