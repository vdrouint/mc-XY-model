#matplotlib inline
from __future__ import division
import numpy as np
from numpy import pi as pi
from numpy import cos as cos
from numpy import sin as sin
from numpy import exp as exp
from numpy import mod as mod
from numpy import absolute as absolute
from numba import jit


######
#-----------------------------------------------------------------------------------------------------------------------
#######
#functions for the Metropolis and Wolff algorithm
######
#-----------------------------------------------------------------------------------------------------------------------
######


#one step of the Wolff algorithm
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def WolffUpdate(config, temp, N, neighbors_list):
    beta=1./temp
    
    #do wolff steps as long as the total cluster size flipped
    # is smaller than the size of the system
    numItTot = N*N
    size = N*N
    #initialize outputs
    avg_size_clust = 0.

    cluster = np.zeros(size, dtype = np.int8)
    listQ = np.zeros(size + 1, dtype = np.int64)

    for nn in range(numItTot):
        #cluster = np.zeros(numItTot, dtype = np.int8)
        #listQ = np.zeros(numItTot + 1, dtype = np.int64)
        init = np.random.randint(0, size)
        listQ[0] = init + 1
        theta_rand = (np.pi)*np.random.rand()   #angle of p*pi, here p is 3
        random_angle =  theta_rand
        
        cluster[init] = 1 #this site is in the cluster now
        sc_in = 0
        sc_to_add = 1

        while listQ[sc_in] != 0:
            site_studied = listQ[sc_in] + (-1)
            sc_in += 1
            avg_size_clust += 1
                
            prime_layer_rand = random_angle
            site_angle = config[site_studied]  #find the current angle of the studied site
            config[site_studied] = (2*prime_layer_rand - site_angle) #site selected has been flipped
 
            for kk in range(4):
                site_nn = neighbors_list[4*site_studied + kk]
                near_angle = config[site_nn]
                if cluster[site_nn] == 0:
                    energy_difference = (-1)*(cos(site_angle - near_angle) - cos(site_angle - (2*prime_layer_rand - near_angle)))
                    freezProb_next = 1. - exp(beta*energy_difference)
                    if (np.random.rand() < freezProb_next):
                        #listQ.append(site_nn)
                        listQ[sc_to_add] = site_nn + (1)                    
                        cluster[site_nn] = 1
                        sc_to_add += 1
        listQ[:] = 0
        cluster[:] = 0

    #average size cluster
    avg_size_clust = avg_size_clust/numItTot    

    return avg_size_clust




#one step of the Wolff algorithm
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def EnergyCalc(config, N):
    energy = 0.
    #calculate the energy
    ####
    for i in range(N):
        for j in range(N):
            latt1 = config[N*i + j]
            latt1shiftX = config[N*(i-1) + j]
            latt1shiftY = config[N*i + j-1]
            energy += (-1.0)*(cos(latt1+(-1)*latt1shiftX) + \
                                  cos(latt1+(-1)*latt1shiftY))
    return energy


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def MeasureConfigNumba(config, N):
    #######
    #all calculations ----------
    #######  
    tpi = 2*pi

    config_re = config
    config_re = config_re.reshape((N,N))
    mod_latt = mod(config_re, tpi)
 

    energy = 0.
    H_tot = 0.
    I_tot = 0.

    #total
    total = 0.
    #bond average
    bond_avg = 0.
    #ord
    ord1 = 0.

    
    #U(1) correlation length
    ordU_xi_x = 0.
    ordU_xi_y = 0.
    
    #some vortex in there
    vort = 0.
    
    for i in range(N):
        for j in range(N):
            platt1 = config_re[i,j]
            platt1shiftX = config_re[i-1,j]
            platt1shiftY = config_re[i,j-1]
            platt1shiftXshiftY = config_re[i-1,j-1]
            
            vcos = cos(platt1+(-1)*platt1shiftX)
            energy += (-1.0)*(vcos + \
                                  cos(platt1+(-1)*platt1shiftY))
            H_tot += vcos
            I_tot += sin(platt1+(-1)*platt1shiftX)

            ord1 += exp(1j*platt1)
          
            #U(1) correlation length
            ordU_xi_x += cos(platt1)*exp(1j*i*tpi/N)
            ordU_xi_y += sin(platt1)*exp(1j*i*tpi/N)
            
            #vortex calc
            platt1v = mod(platt1, tpi)
            platt1shiftXv = mod(platt1shiftX, tpi)
            platt1shiftXshiftYv = mod(platt1shiftXshiftY, tpi)
            platt1shiftYv = mod(platt1shiftY,tpi)
            diff_list1 = np.array([platt1v - platt1shiftXv, platt1shiftXv - platt1shiftXshiftYv,\
                                   platt1shiftXshiftYv - platt1shiftYv, platt1shiftYv - platt1v])
            
            
            vort_here = 0.
            vort_here = 0.
            for ll_1 in diff_list1:
                if ll_1 > np.pi:
                    ll_1 = ll_1 - tpi
                if ll_1 < -np.pi:
                    ll_1 = ll_1 + tpi   
                ll_1 = ll_1/tpi
                vort_here += ll_1
            
                    
            vort += absolute(vort_here)

    vort = vort/tpi
    
    
    #get the norm of these things
    #U(1) correlation length
    ordU_xi_m = (ordU_xi_x.real)**2 + (ordU_xi_x.imag)**2 + (ordU_xi_y.real)**2 + (ordU_xi_y.imag)**2
    

    #now, pack as one list

    #total output
    #to_return = [output_thermo, output_stiff, output_vortex]
    all_dat = np.array([energy, ord1.real, ord1.imag, ordU_xi_m,\
                       H_tot, I_tot, vort])
    #len of 25

    return all_dat