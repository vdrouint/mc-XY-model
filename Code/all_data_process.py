#matplotlib inline
from __future__ import division
import numpy as np
from numpy.random import rand
from numpy import linalg as LA
import time
import sys
from itertools import chain
import os
from numba import jit
#from scipy.optimize import curve_fit

#####
#function for curve fit
#####
def func(x, c, a):
    return (1-a)*np.exp(-c*x) + a

def func2(x, c):
    return np.exp(-c*x)


#####
#jackknife function
#####
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def jackBlocks(original_list, num_of_blocks, length_of_blocks):
    block_list = np.zeros(num_of_blocks)
    length_of_blocks = int(length_of_blocks)
    for i in range(num_of_blocks):
        block_list[i] = (1/length_of_blocks)*np.sum(original_list[i*(length_of_blocks) : (i + 1)*(length_of_blocks)])
    return block_list

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def JackknifeError(blocks, length_of_blocks):
    #blocks is already O_(B,n)
    blocks = np.array(blocks)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def JackknifeErrorFromFullList(original_list, num_of_blocks, length_of_blocks):
    blocks = jackBlocks(original_list, num_of_blocks, length_of_blocks)

    #blocks is already O_(B,n)
    N_B = len(blocks)
    avg = np.sum(blocks)/N_B
    #length_of_blocks is k
    N_J = N_B*length_of_blocks #is basically N
    jack_block = (1/(N_J - length_of_blocks))*(N_J*np.ones(N_B)*avg - length_of_blocks*blocks)
    bar_o_j = np.sum(jack_block)/N_B
    error_sq = ((N_B - 1)/N_B)*np.sum((jack_block - bar_o_j*np.ones(N_B))**2)

    return avg, np.sqrt(error_sq)

########-------------------------------
#Autocorrelation functions
########-------------------------------

def BinningLevel(vals):
    new_vals = np.zeros(len(vals)/2)
    for i in range(len(new_vals)):
        new_vals[i] = (vals[2*i]+vals[2*i+1])/2
    return new_vals


def autocorrelation(data_box, K_max):
    avg_length = len(data_box)
    stack = data_box
    Q_val = np.zeros(K_max)
    for l in range(K_max):
        temp_val = 0.
        num_val = 0
        boolean = True

        while boolean == True:
            l_top = num_val + l
            temp_val += stack[num_val]*stack[l_top]
            
            if l_top == avg_length - 1:
                boolean = False

            num_val += 1

        Q_val[l] = temp_val/num_val

    val_avg = np.sum(stack)/len(stack) 

    #denominator = Q_val[0] - val_avg**2
    #nominator = Q_val - np.ones(len(Q_val))*(val_avg**2)

    #corrData = (1/denominator)*nominator
    return Q_val, val_avg

def autocorrelationBis(data_box, K_max):
    avg_length = len(data_box)
    stack = data_box
    Q_val = np.zeros(K_max)
    Q_err = np.zeros(K_max)
    for l in range(K_max):
        temp_val = []
        num_val = 0
        boolean = True

        while boolean == True:
            l_top = num_val + l
            temp_val.append(stack[num_val]*stack[l_top])
            
            if l_top == avg_length - 1:
                boolean = False

            num_val += 1

        temp_val_avg = np.sum(temp_val)/len(temp_val)
        temp_val_err = np.std(temp_val)

        Q_val[l] = temp_val_avg
        Q_err[l] = temp_val_err


    return Q_val, Q_err

#K : length of the stack array for the values
#avg_length : the number of evaluations taken for the averages of the stack
#error _length : number of times it is computed to get error bars
def mainPartAutocorrelation(data, K, avg_length, error_length):
    
    full_data = []
    average_file = []
    num_of_elements = np.array([(avg_length - i) for i in range(K)])

    for i in range(error_length):
        Q_val, val_avg  = autocorrelation(data[avg_length*i:avg_length*(i+1)], K)
        full_data.append(Q_val) #has all the list of averages Q_t Q_t+tau
        average_file.append(val_avg) #has the average of Q_t
    full_data = np.array(full_data)
    average_file = np.array(average_file)
    
    #Jackknife error analysis
    #see function
    #error on <o> 
    avg_o, error_o = JackknifeError(average_file, error_length)
    #error on <o_i o_i+T>
    avg_ooT = np.zeros(K)
    error_ooT = np.zeros(K)
    for m in range(K):
        avg_ooT[m], error_ooT[m] = JackknifeError(full_data[:,m], num_of_elements[m])

    A_corr_denom = avg_ooT[0] - avg_o**2
    A_corr_nom = avg_ooT - np.ones(K)*(avg_o**2)

    A_corr = (1/(A_corr_denom))*(A_corr_nom) 
    error_A_corr_denom = np.sqrt((error_ooT[0])**2 + (2*np.fabs(avg_o)*error_o)**2)
    error_A_corr_nom = np.sqrt((error_ooT)**2 + np.ones(K)*(2*np.fabs(avg_o)*error_o)**2)
    error_A_corr = np.fabs(A_corr)*np.sqrt((np.divide(error_A_corr_nom,A_corr_nom))**2 + (np.divide(error_A_corr_denom,A_corr_denom))**2)

    return A_corr, error_A_corr

def mainPartAutocorrelationBis(data, K_max):
    #avg and error on <o_i o_i+T>
    avg_ooT, error_ooT  = autocorrelationBis(data, K_max)
    #print np.absolute(np.divide(error_ooT, avg_ooT))[:6]
    #avg and error on the average of O, <o>
    avg_o = np.sum(data)/len(data)
    error_o = np.std(data)

    #print np.absolute(np.divide(error_o, avg_o))

    A_corr_denom = avg_ooT[0] - avg_o**2
    A_corr_nom = avg_ooT - np.ones(K_max)*(avg_o**2)

    A_corr = (1/(A_corr_denom))*(A_corr_nom) 
    error_A_corr_denom = np.sqrt((error_ooT[0])**2 + (2*np.fabs(avg_o)*error_o)**2)
    #print A_corr_denom, error_A_corr_denom
    error_A_corr_nom = np.sqrt((error_ooT)**2 + np.ones(K_max)*((2*np.fabs(avg_o)*error_o)**2))
    #error_A_corr_denom = np.sqrt((error_ooT[0])**2 + (2*np.fabs(avg_o)*error_o)**2)
    #error_A_corr_nom = np.sqrt((error_ooT)**2 + np.ones(K_max)*(2*np.fabs(avg_o)*error_o)**2)
    #print np.absolute(np.divide(error_A_corr_denom, A_corr_denom))
    #print np.absolute(np.divide(error_A_corr_nom, A_corr_nom))[:6]
    error_A_corr = np.fabs(A_corr)*np.sqrt((np.divide(error_A_corr_nom,A_corr_nom))**2 + (np.divide(error_A_corr_denom,A_corr_denom))**2)

    #print A_corr[:6]
    #print error_A_corr[:6]

    return A_corr, error_A_corr


#Need to first extract the zip file
#the format the string files
def main():

    ######
    #-----------------------------------------------------------------------------------------------------------------------
    #######
    #parameters of the code
    ######
    #-----------------------------------------------------------------------------------------------------------------------
    ######

    N = int(sys.argv[1])  #note that N is in fact L the linear size
    factor_print = 10000
    jint = 1.0

    L_size = N

    ##########
    #when you do it right after computation 
    ##########

    #the directory to take the data from 
    name_dir = 'testL='+str(int(N)) 

    #folder to put it in
    folder_data_final = name_dir+'finalData'
    if not os.path.exists('../Results/'+ folder_data_final):
        os.mkdir('../Results/'+ folder_data_final)

    #saving the variables in the other folder
    #the form of variables.data
    #saving_variables_pre = np.array([j2,j6,lambda3, Kc, length_box, number_box, therm])
    #saving_variables = np.append(saving_variables_pre, range_temp)
    
    saving_variables = np.loadtxt('./'+name_dir+'/variables.data')
    number_box = int(saving_variables[1])
    #number_box = 1600
    length_box = int(saving_variables[0])
    #length_box = 10
    range_temp = saving_variables[3:]
    np.savetxt('../Results/'+folder_data_final+'/variables.data', saving_variables)
    #number of temperature steps
    nt = len(range_temp)


    ########
    #Processing the data in order to be plotted
    ########

    """
    #originally came as
    all_dat = np.array([energy, ord1.real, ord1.imag, ordU_xi_m,\
                       H_tot, I_tot, vort])

    #order of data
    #thermo
    output_thermo = [energy, ord1.real, ord1.imag, ordU_xi_m]
    #len of 4
      

    #stiffness
    output_stiff = [H_tot, I_tot]
    #len of 2

    #vortex
    output_vortex = [vort]
    #len of 1

    """

    print()
    print('Starting Analysis')
    print()

    

    #########-----------------------------------------------
    #Measurements for the energy and ~magnetization
    ########------------------------------------------------

    Energy = np.zeros(2*nt)
    SpecificHeat = np.zeros(2*nt)
    OrderCumulant = np.zeros(2*nt)

    number_bins_ene = int(number_box*length_box/20)
    Energy_histo = np.zeros((nt, number_bins_ene))
    Energy_histo_edges = np.zeros((nt, number_bins_ene + 1))

    OrderParam = np.zeros(2*nt)
    OrderParam_BIS = np.zeros(2*nt)
    Susceptibility1 = np.zeros(2*nt)
    Susceptibility2 = np.zeros(2*nt)
    BinderCumulant = np.zeros(2*nt)

    CorrLengthU = np.zeros(2*nt)

    ## This part runs through the data and creates the errors
    for m in range(nt):
        data = np.loadtxt('./' + name_dir +'/outputatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        #data = np.loadtxt('outputL='+str(int(N))+'atT='+str(int(range_temp[m]*1000)).zfill(5)+'.data')
        """
        output_thermo = [energy, total.real, total.imag, ord6.real, ord6.imag,\
        ord2.real, ord2.imag, tot_sector.real, tot_sector.imag,\
        cm_order.real, cm_order.imag, locking, bond_avg,\
        ord6_xi_m, ord2_xi_m, ordP_xi_m, ordU_xi_m]
        """
        
        E1_init = np.divide(np.array(data[:,0]), N**2)
        M1_init = np.divide(np.array(data[:,1]) + 1j*np.array(data[:,2]), N**2)
        #corr_length prep
        ordU_xi = np.divide(np.array(data[:,3]), N**4) 

        ordU_xi_0 = np.divide(np.array(data[:,1])**2 + np.array(data[:,2])**2, N**4) 
        
        #energy
        E1 = E1_init
        E2 = E1*E1
        E4 = E2*E2
        
        #magnetization variable
        M1_real = np.real(M1_init) # only avg of cos
        M1_imag = np.imag(M1_init) # only avg of cos
        M1_tot = np.absolute(M1_init)
        M2 = np.absolute(M1_init)**2
        M4 = M2*M2


        #correlation length are all fine


        #we use a version of the jackknife function that creates the boxes in the function
        E1_avg , E1_error = JackknifeErrorFromFullList(E1, number_box, length_box)
        E2_avg , E2_error = JackknifeErrorFromFullList(E2, number_box, length_box)
        E4_avg , E4_error = JackknifeErrorFromFullList(E4, number_box, length_box)
        #all order
        M1_real_avg , M1_real_error = JackknifeErrorFromFullList(M1_real, number_box, length_box)
        M1_imag_avg , M1_imag_error = JackknifeErrorFromFullList(M1_imag, number_box, length_box)
        M1_avg, M1_error = JackknifeErrorFromFullList(M1_tot, number_box, length_box)
        M2_avg , M2_error = JackknifeErrorFromFullList(M2, number_box, length_box)
        M4_avg , M4_error = JackknifeErrorFromFullList(M4, number_box, length_box)
        

        #correlation length
        ordU_xi_avg, ordU_xi_err = JackknifeErrorFromFullList(ordU_xi, number_box, length_box)
        ordU_xi_0_avg, ordU_xi_0_err = JackknifeErrorFromFullList(ordU_xi_0, number_box, length_box)

        #energy related observables
        #E, Cv
        Energy[m]         = E1_avg
        Energy[nt + m]         = E1_error
        div_sp = (range_temp[m]**2)
        SpecificHeat[m]   = ( E2_avg - E1_avg**2)/div_sp 
        SpecificHeat[nt + m]   = (((E2_error)**2 + (2*E1_error*E1_avg)**2)**(0.5))/div_sp
        ord_cum = (E4_avg)/(E2_avg**2) - 1
        OrderCumulant[m] = ord_cum
        OrderCumulant[nt + m] = np.fabs(ord_cum)*np.sqrt((E4_error/E4_avg)**2 + (2*E2_error/E2_avg)**2)

        #locking related observables
        #|<m>|, <|m|>, chi1 = (<m^2> - <|m|>^2)/T, chi2 = (<m^2>)/T, binder
        u_op = M1_real_avg**2 + M1_imag_avg**2
        u_op_err = np.sqrt((2*M1_real_error*M1_real_avg)**2 + (2*M1_imag_error*M1_imag_avg)**2)
        OrderParam[m]  = np.sqrt(u_op)
        OrderParam[nt + m] = 0.5*u_op_err/np.sqrt(u_op)
        OrderParam_BIS[m] = M1_avg
        OrderParam_BIS[nt + m] = M1_error
        Susceptibility1[m] = ( M2_avg - M1_avg**2)/(range_temp[m]);
        Susceptibility1[nt + m] = np.sqrt((M2_error)**2 + (2*M1_avg*M1_error)**2)/(range_temp[m]);
        Susceptibility2[m] = ( M2_avg)/(range_temp[m]);
        Susceptibility2[nt + m] = ( M2_error)/(range_temp[m]);  
        bind_cum = M4_avg/(M2_avg**2)
        BinderCumulant[m] = 1 - bind_cum/3
        BinderCumulant[nt + m] = (1/3)*np.fabs(bind_cum)*np.sqrt((M4_error/M4_avg)**2 + (2*M2_error/M2_avg)**2) 

        
        #######
        #put the correlation length
        #######
        #corr length
        fact_sin = (1/(2*np.sin(np.pi/N))**2)
        val_U_c = ((ordU_xi_0_avg/ordU_xi_avg) -1)
        CorrLengthU[m] = fact_sin*val_U_c
        CorrLengthU[nt + m] = fact_sin*(np.sqrt((ordU_xi_0_err/ordU_xi_0_avg)**2 \
            + (ordU_xi_err/ordU_xi_avg)**2))
       


        ########
        #do a temperature histogram
        #######
        min_energy = np.min(E1)
        max_energy = np.max(E1)
        bound_low = min_energy + 0.01*(max_energy - min_energy)
        bound_high = max_energy - 0.01*(max_energy - min_energy)
        ener_histo = np.histogram(E1, number_bins_ene, range = (bound_low, bound_high))
        Energy_histo[m] = ener_histo[0]
        Energy_histo_edges[m] = ener_histo[1]


    #save an energy histogram
    #np.savetxt('./'+ folder_data_final +'/histo_output.data', np.c_[Energy_histo])
    #np.savetxt('./'+ folder_data_final +'/histo_edges_output.data', np.c_[Energy_histo_edges])



    #np.savetxt('./testF=1L='+str(int(N))+'finalData'+'/thermo_outputL='+str(int(N))+'.data',
    #    np.c_[Energy, SpecificHeat, OrderCumulant, OrderParam, Susceptibility, BinderCumulant, \
    #    Order6, Susc6,Binder6, Order2,  Susc2, Binder2, MomentumNullHexatic, MomentumNullNematic])

    np.savetxt('../Results/'+ folder_data_final +'/thermo_output.data',
        np.c_[Energy, SpecificHeat, OrderCumulant,\
        OrderParam, OrderParam_BIS, Susceptibility1, Susceptibility2, BinderCumulant, CorrLengthU])

    print()
    print('Done with Order/Thermo Analysis')
    print()



    #########-----------------------------------------------
    #Measurements for the Stiffness
    ########------------------------------------------------

    RhoTot = np.zeros(2*nt)

    fourthOrderTot = np.zeros(2*nt)


    ## This part runs through the data and creates the errors
    for m in range(nt):
        data = np.loadtxt('./'+ name_dir +'/stiffnessPreDataatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        #data = np.loadtxt('stiffnessPreDataL='+str(int(N))+'atT='+str(int(range_temp[m]*1000)).zfill(5)+'.data')

        Stiff_tot_H = data[:,0]/N**2
        Stiff_tot_I = data[:,1]/N**2
        Stiff_tot_I2 = Stiff_tot_I*Stiff_tot_I
        Stiff_tot_I4 = Stiff_tot_I2*Stiff_tot_I2

        #we use a version of the jackknife function that creates the boxes in the function        
        Stiff_tot_H_avg , Stiff_tot_H_error = JackknifeErrorFromFullList(Stiff_tot_H, number_box, length_box)
        Stiff_tot_I_avg , Stiff_tot_I_error = JackknifeErrorFromFullList(Stiff_tot_I, number_box, length_box)
        Stiff_tot_I2_avg , Stiff_tot_I2_error = JackknifeErrorFromFullList(Stiff_tot_I2, number_box, length_box)
        Stiff_tot_I4_avg , Stiff_tot_I4_error = JackknifeErrorFromFullList(Stiff_tot_I4, number_box, length_box)

        T = range_temp[m]    
        RhoTot[m] = Stiff_tot_H_avg - (N**2/T)*(Stiff_tot_I2_avg - Stiff_tot_I_avg**2)
        RhoTot[nt + m] = np.sqrt(Stiff_tot_H_error**2 + ((N**2)*Stiff_tot_I2_error/T)**2 + ((N**2)*2*Stiff_tot_I_error*Stiff_tot_I_avg/T)**2 )

        #avg <(Y-<Y>)^2> = <Y^2> - <Y>^2
        list_Ysq_tot = (Stiff_tot_H - (N**2/T)*(Stiff_tot_I2 - Stiff_tot_I**2))**2
        list_Ysq_tot_avg, list_Ysq_tot_error = JackknifeErrorFromFullList(list_Ysq_tot, number_box, length_box)

        #here we compute <L**2 Y_4>
        fourthOrderTot[m] = (-1)*4*RhoTot[m] + 3*(Stiff_tot_H_avg\
            - (N**2/T)*(list_Ysq_tot_avg - RhoTot[m]**2)) + 2*(N**6/(T**3))*Stiff_tot_I4_avg


    np.savetxt('../Results/'+ folder_data_final +'/STIFF_thermo_output.data',
        np.c_[RhoTot, fourthOrderTot])

    print()
    print('Done with Stiffness Analysis')
    print()


    #########-----------------------------------------------
    #Measurements for the Vorticity
    ########------------------------------------------------

    Vorticity = np.zeros(2*nt)
    

    #vorticity measurements
    for m in range(nt):
        data = np.loadtxt('./'+ name_dir +'/VorticityDataatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
        Vort = data[:]
        #we use a version of the jackknife function that creates the boxes in the function
        Vort_avg , Vort_error = JackknifeErrorFromFullList(Vort, number_box, length_box)
        Vorticity[m] = Vort_avg/N**2
        Vorticity[nt + m] = Vort_error/N**2


    #np.savetxt('./'+ folder_data_final +'/Vorticity_thermo_output.data',
    #    np.c_[VorticityTheta, VorticityPhi, DeviationTheta, DeviationPhi, diffTheta, diffPhi, fracVortexPhi])
    np.savetxt('../Results/'+ folder_data_final +'/Vorticity_thermo_output.data',
        np.c_[Vorticity])

    print()
    print('Done with Vorticity Analysis')
    print()

    
    #this creates a larger file
    if False:
        ###########
        #Measurement of the autocorrelation
        ###########

        ####
        #length of autocor analysis
        ####
        K_max = length_box
        K_val = np.arange(K_max)

        for m in range(nt):
            data_1 = np.loadtxt('./' + name_dir +'/outputatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
            AutoCorrEnergy = np.zeros(2*K_max)
            AutoCorrM = np.zeros(2*K_max)


            #energy is 0
            #m6 is 3,4
            #m2 is 5,6
            #mrel is 7,8

            energy_to_autocorr = np.array(data_1[:,0])
            auto_ene, auto_ene_err = mainPartAutocorrelationBis(np.divide(energy_to_autocorr, N**2), K_max)
            AutoCorrEnergy = np.concatenate((auto_ene, auto_ene_err))
      
            m_to_autocorr = np.absolute(np.array(data_1[:,1]) + 1j*np.array(data_1[:,2]))
            auto_m, auto_m_err = mainPartAutocorrelationBis(np.divide(m_to_autocorr, N**2), K_max)
            AutoCorrM = np.concatenate((auto_m, auto_m_err))


            ##save it
            np.savetxt('./'+ folder_data_final +'/Autocorr_outputatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data',
                np.c_[AutoCorrEnergy, AutoCorrM])

        print()
        print('Done with Autocorrelation Analysis')
        print()


        #####
        #move config to here
        #####
        for m in range(nt):
            data_1 = np.loadtxt('./' + name_dir +'/configatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data')
            np.savetxt('./'+ folder_data_final +'/configatT='+str(int(range_temp[m]*factor_print)).zfill(5)+'.data',data_1)        


    
#------------------------------------------------
#the executable part
#------------------------------------------------

if __name__ == '__main__':

    main()
