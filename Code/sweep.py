from numpy import arange
import subprocess

L_list = [6, 10, 16, 20, 26, 30, 40] #N
#L_list = [4]

therm = 200  #better 300
meas = 300  #better 300


nt = 8*4
corenum = 8
Tmin = 0.1
Tmax = 1.5

type_of_temp_range = 0

script_val = """script.sh"""   #for i08m3c

for L in L_list:
    lx = L
    lz = L
    pre_command = """qsub -N runL={length} """.format(length=L, cn=corenum)
    values_command = """ -v Lx={d1},Tmin={d2},Tmax={d3},NumberTemp={d4},Type={d5},NumCores={d6},Therm={d7},Measure={d8}  """.format(\
        d1 = lx, d2 = Tmin, d3 = Tmax, d4 = nt, d5 = type_of_temp_range, d6 = corenum, d7=therm, d8=meas)
    qsub_command = pre_command + values_command + script_val

    #print qsub_command # Uncomment this line when testing to view the qsub command

    # Comment the following 3 lines when testing to prevent jobs from being submitted
    subprocess.call(qsub_command, shell=True)

