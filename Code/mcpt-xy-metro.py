#matplotlib inline
from __future__ import division, print_function
import numpy as np
from numpy.random import rand
import time
import sys
import os
#import zipfile
#from memory_profiler import profile
from joblib import Parallel, delayed
import math
from numpy import pi as pi
from numpy import cos as cos
from numpy import sin as sin
from numpy import exp as exp
from numpy import mod as mod
from numpy.random import randint as randint
from numpy import absolute as absolute
#this is where you import the MC update
from functions_mcstep import MetropolisUpdate as mcUpdate
from functions_mcstep import EnergyCalc as EnergyCalc
from functions_mcstep import MeasureConfigNumba as MeasureConfigNumba



####
#No pre thermalization in Metropolis Step
####


####
#thermalization step
####

def PTTstepTherm(config_init, temp, N, neighbors_list, factorIter):

    #the config
    config = config_init.copy()

    energies = np.zeros(factorIter)

    for st in range(factorIter):
        mcUpdate(config, temp, N, neighbors_list)
        energies[st] = EnergyCalc(config, N)

    #final energy
    energy = energies[-1]

    return [config, energy, np.array(energies)]

#####
#measurement step
#####

def PTTstepMeasure(config_init, temp, N, neighbors_list, factorIter):

    #the config
    config = config_init.copy()

    data_thermo = []
    for st in range(factorIter):
        mcUpdate(config, temp, N, neighbors_list)
        data_thermo.append(MeasureConfigNumba(config, N))

    #final energy
    energy = data_thermo[-1][0]

    return [config, energy, np.array(data_thermo)]



##############-------------------------------------------------------------
#Main function that runs the MC code
##############-------------------------------------------------------------

#@profile
def main():

    N = int(sys.argv[1])  #note that N is in fact L the linear size
    factor_print = 10000
    jint = 1.0

    L_size = N

    #temp range
    Tmin = float(sys.argv[2])
    Tmax = float(sys.argv[3])
    nt = int(sys.argv[4])
    type_of_temp_range = int(sys.argv[5])  #either 0 (geometric) or 1 (linear)

    #number of steps
    num_cores = int(sys.argv[6])
    length_box = 100 # number of MC steps in each bin (both during measurement and during thermalization period)   
    #pre_therm = 100     
    therm = int(sys.argv[7])     # number of MC bins during thermalization time
    number_box = int(sys.argv[8])     # // number of MC bins during which measurement is applied 
    

    #beta range
    #then figure out T range
    #but the adaptative step will be on beta
    Beta_min = 1/Tmax
    Beta_max = 1/Tmin

    #the list of temperatures and the list of energies, initialized
    if type_of_temp_range == 0:
        #geometric temp range
        ratio_T = (Tmax/Tmin)**(1/(nt-1))
        range_temp = np.zeros(nt)
        for i in range(nt):
            range_temp[i]=Tmin*((ratio_T)**(i))
    elif type_of_temp_range == 1:
        range_temp = np.linspace(Tmin, Tmax, nt)
    list_temps = range_temp
    list_energies = np.zeros(nt)       

    ######
    #initialize list of neighbors
    ######

    neighbors_list = np.zeros(4*(N**2))
    sizeN = N**2
    for i in range(N**2):
        vec_nn_x = [-1, 1, 0, 0]
        vec_nn_y = [0,  0, -1,1]

        site_studied_1 = i//N
        site_studied_2 = i%N
        for p in range(4):
            neighbors_list[4*i + p] = (N*mod((site_studied_1 + vec_nn_x[p]),N) + mod((site_studied_2 + vec_nn_y[p]),N))
        #neighbors_list[5*i + 4] = ((i + sizeN)%(2*sizeN))

    neighbors_list = np.array(list(map(int, neighbors_list)))
    #print neighbors_list

    #########
    #initialize folder
    ##########

    #see if the folder exists. if it does not, create one
    name_dir = 'testmetroL='+str(int(N)) 

    where_to_save = './' #more useful for cluster use
    #where_to_save = '../Results/' #more useful for local runs

    if where_to_save == '../Results/':
        if not os.path.exists('../Results'):
            os.mkdir('../Results')

    #z = zipfile.ZipFile(name_dir + ".zip", "w")
    if not os.path.exists(where_to_save + name_dir):
        os.mkdir(where_to_save + name_dir)

    #print the initial parameters
    print() 
    print('Linear size of the system L=' + str(N))
    print('Interaction strength:')
    print('J = ' + str(jint))
    print()
    print('From temperature Tmax='+ str(Tmax)+' to Tmin='+str(Tmin))
    print('In '+str(nt)+' steps')
    print()
    print('Size of bins:' + str(length_box))
    #print 'N_inter' + str(N_inter)
    print('Number of thermalization bins:' + str(therm))
    print('Number of measurement bins:' + str(number_box))
    print('number of Cores' + str(num_cores))
    print()

    #initializing the configurations 
    config_start = []
    for q in range(nt):
        config_start.append(2*pi*rand(N**2))
    config_start = np.array(config_start)

    #important definitions for parallel tempering
    #indices of temperatures -> energies (config)
    #indices of energies (config) -> temperatures

    indices_temp = [i for i in range(nt)] #pt_TtoE
    pt_TtoE = indices_temp
    indices_ensemble = [i for i in range(nt)] #pt_EtoT
    pt_EtoT = indices_ensemble


    print('starting the initialization step')
    print()

    start = time.time()

    print('list of temperatures')
    print(list_temps)

    #first run in order to get some estimate of the energy
    #use pt in this run
    print('start with pre therm')
    config_at_T = config_start #the initial config of config_at_T is defined


    #saving the variables of the computation
    saving_variables_pre = np.array([length_box, number_box, therm])
    saving_variables = np.append(saving_variables_pre, list_temps)
    np.savetxt(where_to_save + name_dir +'/variables.data', saving_variables)
    #np.savetxt('./'+ name_dir +'variables.data', saving_variables)

    #-----------------
    #Prep for the Therm + Measure using Parallel Tempering
    #------------


    #list of tuples for the parallel tempering
    tuples_1 = [indices_temp[i:i + 2] for i in range(0, len(indices_temp), 2)] #odd switch #len of nt/2
    tuples_2 = [indices_temp[i:i + 2] for i in range(1, len(indices_temp) - 1, 2)] #even switch #len of nt/2 -1 
    tuples_tot = [tuples_1, tuples_2]  
    half_length = int(nt/2)
    len_tuples_tot = [half_length, half_length - 1]

    ###-------------------------------------------------------------------
    #main program
    #does the MC steps (Metropolis and Wolff) + parallel tempering
    #then measures the energy/mag and spin stiffness
    ###-------------------------------------------------------------------
    #we already have an initial config as config_start
    #and we have a list of initial energies as energy_start

    #we want to keep the measured quantities for mc_data_len steps
    #number_of_parallel_it = length_box*number_box
    mc_data_len = number_box*(length_box)
    #we want to thermalize the system for therm steps
    length = therm*(length_box)

    ###------------------------------------------------------------------
    #start the thermalization
    ###------------------------------------------------------------------


    print()
    print('Starting the thermalization')
    print()
    start = time.time()

    #swap even pairs or not: initiate at 0
    swap_even_pairs = 0
    #the therm procedure
    #opening single threads and not destroying them
    #note that length box
    #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)

    all_the_energies_thermalization = np.zeros((nt,therm*length_box))

    with Parallel(n_jobs=num_cores, max_nbytes = '5M') as parallel:
        for il in range(therm):
            #the - period in the range of length_box is to account for the 'period' steps in optimization part
            #print('gone to  ' + str(int(il)) + ' ' + str(int(jl)))

            #######-------
            #Monte Carlo step 
            #######------------

            #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)
            resultsTherm = parallel(delayed(PTTstepTherm)(config_init = config_at_T[m], \
                temp = list_temps[pt_EtoT[m]], N= N, neighbors_list = neighbors_list, \
                factorIter = length_box) for m in range(nt))
            for q in range(nt):
                list_energies[q] = resultsTherm[q][1]
                config_at_T[q] = resultsTherm[q][0]
                data_extract = resultsTherm[pt_TtoE[q]][2]
                for ws in range(length_box):
                    all_the_energies_thermalization[q][length_box*il + ws] = data_extract[ws] 

            #####---------
            #The Parallel Tempering Step
            #do it after every box is done running
            #####----------

            #tuples to use
            tuples_used = tuples_tot[swap_even_pairs]
            for sw in range(len_tuples_tot[swap_even_pairs]):
                index_i = tuples_used[sw][0]
                index_j = tuples_used[sw][1]
                initial_i_temp = list_temps[index_i]
                initial_j_temp = list_temps[index_j]
                index_energy_i = pt_TtoE[index_i]
                index_energy_j = pt_TtoE[index_j]

                Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j])*(1/initial_i_temp - 1/initial_j_temp)
                if Delta_ij > 0:
                    pt_TtoE[index_i] = index_energy_j
                    pt_TtoE[index_j] = index_energy_i  
                    pt_EtoT[index_energy_i] = index_j
                    pt_EtoT[index_energy_j] = index_i
                else:                  
                    if rand() < exp(Delta_ij):
                        pt_TtoE[index_i] = index_energy_j
                        pt_TtoE[index_j] = index_energy_i  
                        pt_EtoT[index_energy_i] = index_j
                        pt_EtoT[index_energy_j] = index_i

            #change the pair swapper for next run
            swap_even_pairs = (1 - swap_even_pairs)
            #gc.collect()
            print('Done with therm step' + str(int(il)) + ' out of ' + str(int(therm)) )

    end = time.time()
    #done with thermalization
    print()
    print('Done with thermalization')
    print('in '+str(end - start)+' seconds')
    print('number of steps ' + str(length))
    print('time per step (full PT + Metro)' + str((end - start)/(length) ))
    print('time per step per temp (full PT + Metro)' + str((end - start)/(length*nt) ))
    print()

    print('a comparison of the energies for the first and last quarter of therm')
    print()

    #showing you the energues binned through, 3rd quarter vs 4th quarter
    #is the average energy very similar?
    for q in range(nt):
        startv1 = int(length_box*therm/2)
        endv1 = int(3*length_box*therm/4)
        startv2 = endv1
        endv2 = int(length_box*therm)
        print(range_temp[q])
        print('first average')
        dat1 = all_the_energies_thermalization[q][startv1:endv1]/(N*N)
        print(np.mean(dat1), np.std(dat1))
        print('second average')
        dat2 = all_the_energies_thermalization[q][startv2:endv2]/(N*N)
        print(np.mean(dat2), np.std(dat2))
        print('difference ', np.mean(dat1) - np.mean(dat2))
        print()

    print('now onto the measurement!')
    print()       

    print()
    print('Starting the measurements')
    print()
    start = time.time()

    ###------------------------------------------------------------------
    #start the measurements
    ###------------------------------------------------------------------

    

    #the data sets:
    #change this
    all_data_thermo = np.zeros((nt,number_box*length_box, 7))

    #swap even pairs or not: initiate at 0
    swap_even_pairs = 0
    #the therm procedure
    #opening single threads and not destroying them



    with Parallel(n_jobs=num_cores, max_nbytes = '5M') as parallel:
        for il in range(number_box):

            #the - period in the range of length_box is to account for the 'period' steps in optimization part

            #######-------
            #Monte Carlo step 
            #######------------

            #run all configs at a given temperature, use pt_EtoT to get the right temperature (E is like Config)
            resultsMeasure = parallel(delayed(PTTstepMeasure)(config_init = config_at_T[m], \
                temp = list_temps[pt_EtoT[m]], N= N, neighbors_list = neighbors_list, \
                factorIter = length_box) for m in range(nt))
            for q in range(nt):
                list_energies[q] = resultsMeasure[q][1]
                config_at_T[q] = resultsMeasure[q][0]

            #####---------
            #The Parallel Tempering Step
            #####----------

            #tuples to use
            tuples_used = tuples_tot[swap_even_pairs]
            for sw in range(len_tuples_tot[swap_even_pairs]):
                index_i = tuples_used[sw][0]
                index_j = tuples_used[sw][1]
                initial_i_temp = list_temps[index_i]
                initial_j_temp = list_temps[index_j]
                index_energy_i = pt_TtoE[index_i]
                index_energy_j = pt_TtoE[index_j]

                Delta_ij = (list_energies[index_energy_i] - list_energies[index_energy_j])*(1/initial_i_temp - 1/initial_j_temp)
                if Delta_ij > 0:
                    pt_TtoE[index_i] = index_energy_j
                    pt_TtoE[index_j] = index_energy_i  
                    pt_EtoT[index_energy_i] = index_j
                    pt_EtoT[index_energy_j] = index_i
                else:                  
                    if rand() < exp(Delta_ij):
                        pt_TtoE[index_i] = index_energy_j
                        pt_TtoE[index_j] = index_energy_i  
                        pt_EtoT[index_energy_i] = index_j
                        pt_EtoT[index_energy_j] = index_i

            #change the pair swapper for next run
            swap_even_pairs = (1 - swap_even_pairs)

            #reading the data and saving it
            #note that I want a given column of these data set to a distinct temperature, so I use pt_TtoE in there.            
            for q in range(nt):
                data_extract = resultsMeasure[pt_TtoE[q]][2]
                for ws in range(length_box):                 
                    all_data_thermo[q][length_box*il + ws] = data_extract[ws]

            print('Done with measure step' + str(int(il)) + ' out of ' + str(int(number_box)) )


    end = time.time()
    #done with measurements
    print()  
    print('Done with measurements')
    print('in '+str(end - start)+' seconds')
    print('number of PT steps ' + str(mc_data_len))
    print('time per step (full PT + Metro)' + str((end - start)/(mc_data_len) ))
    print('time per step per temp (full PT + Metro)' +str((end - start)/(mc_data_len*nt) )  )
    print ()

    ############
    #-------------------------
    ############
    #Exporting the data
    ############
    #------------------------
    ############

    #export data
    #define folder

    for q in range(nt):
        temp_init = list_temps[q]
        np.savetxt(where_to_save + name_dir +'/configatT='+str(int(temp_init*factor_print)).zfill(5)+'.data',config_at_T[pt_TtoE[q]])
        np.savetxt(where_to_save + name_dir +'/outputatT='+str(int(temp_init*factor_print)).zfill(5)+'.data',all_data_thermo[q])

       
    print() 
    print('Done with exporting data')
    

#------------------------------------------------
#the executable part
#------------------------------------------------

if __name__ == '__main__':

    main()
