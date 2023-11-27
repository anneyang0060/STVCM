# Code for STVCM

### Overview

Directory ***temporal*** and ***Spatio-temporal*** contain the main functions for numerical experiments for temporal and spatio-temporal varying coefficients models, respectively. 


### Workflows

#### Simulation for temporal varying coefficients models

1. Make sure that ***model_new.py*** is in the folder ***temporal/src*** and ***V2_cityA_serial_order_dispatch_AA.csv***, ***V2_cityB_serial_order_dispatch_AA.csv***  are in the folder ***temporal/data***. Set the file names to *V2_cityA_serial_order_dispatch_AA.csv* and *V2_cityB_serial_order_dispatch_AA.csv* respectively, and follow the steps below 
2. Run ***temporal/generate Fig 1.ipynb*** to generate Figure 1 in the main context.
3. Run ***temporal/sim/model_test_new.py*** to run the main simulation and generate the simulated p-values for DE and IE tests. The results are stored in the folder ***temporal/res***.
4. Note that **Figure 2-3** are plotted based on **Table S1-S4**. Run ***generate table S1-S4.ipynb*** to generate these tables.  

#### Simulation for Spatio-temporal varying coefficients models

1. Make sure that ***model_st_new.py*** is in the folder ***Spatio-temporal/src*** and ***V1_CityF_pool.csv*** is in the folder ***Spatio-temporal/data***.
2. Run ***Spatio-temporal/generate Fig 4.ipynb*** to generate Figure 4 in the main context.
3. Run ***Spatio-temporal/sim/model_test_st_new.py*** to run the main simulation and generate the simulated p-values for DE and IE tests. The results are stored in the folder ***Spatio-temporal/res***.
4. Note that **Figure 5-6** are plotted based on **Table S5-S6**. Run ***generate table S5-6.ipynb*** to generate these tables.  

#### Data application

The real data used in the main context are proprietary and cannot be shared. Hence we provide simulated data sets to which can generate similar results instead. These data are in the folders *temporal/data** and *Spatio-temporal/data*. Run ***temporal/realData_tem.ipynb*** to obtain similar results as Table 1-2 and Figure 7-8. Run ***Spatio-temporal/realData_st.ipynb*** to obtain similar results as Table 3 and Figure 9-10. 

