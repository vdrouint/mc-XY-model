from numpy import arange
import subprocess

N_list = [20, 40, 60, 80, 100] #N
j2_list = [1.0] #j2
ll_list = [2.1] #lambda
for job in N_list:
    for job2 in j2_list:
        for ll in ll_list:

            #node = """wp48m4@n137"""
            node = """wp08m3c"""
            corenum = 8
            nt = 16
            Tmin = 1.17
            Tmax = 1.22


            pre_command = """qsub -N mcPT{first}l{fourth} -q """.format(first=job,, fourth = ll) + node + """ -pe mpi2 {cores} -v """.format(cores = corenum)
            qsub_command = pre_command + """N={first},J2={second},Lambda={fourth},NumberTemp={fifth},NumCores={cores},Tmin={Tmin},Tmax={Tmax} script2N.sh""".format(\
                cores = corenum, first=job, \
                second=job2, fourth = ll, fifth = nt, Tmin = Tmin, Tmax = Tmax)

            #print qsub_command # Uncomment this line when testing to view the qsub command

            # Comment the following 3 lines when testing to prevent jobs from being submitted
            subprocess.call(qsub_command, shell=True)

