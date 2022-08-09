import opendssdirect as dss
from opendssdirect.utils import run_command
import pandas as pd
import os

dir_to_feeder=os.getcwd()
dir_to_results = os.path.join(dir_to_feeder,'simulation_results')

run_command('clear')
run_command('compile IEEE34bus/ieee34_test.dss')
run_command("set mode=snapshot")
dss.Solution.ControlMode(1)

print(f'Circuit power demand is: {dss.Circuit.TotalPower()}')

run_command('New generator.der1 bus1=850.2 Phases=1 Model=1 Conn=Wye kV=14.376 kW=50 kVAR=0')
run_command('New generator.der2 bus1=818.1 Phases=1 Model=1 Conn=Wye kV=14.376 kW=52.5 kVAR=0')
run_command('New generator.der3 bus1=830.1 Phases=1 Model=1 Conn=Wye kV=14.376 kW=50 kVAR=0')
run_command('New generator.der4 bus1=844.2 Phases=1 Model=1 Conn=Wye kV=14.376 kW=50 kVAR=0')

run_command('solve')

print(f'Circuit power demand is: {dss.Circuit.TotalPower()}')

voltage_mag = dss.Circuit.AllBusVolts()
voltage_pu = dss.Circuit.AllBusMagPu()

pu_df = pd.DataFrame(voltage_pu)
mag_df=pd.DataFrame(voltage_mag)

pu_df.to_csv(dir_to_results+'\\voltage_pu'+'.csv')
mag_df.to_csv(dir_to_results+'\\voltage_mag'+'.csv')


