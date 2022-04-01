#!/bin/bash
#set -x
########################################################################
# SUN Grid Engine job wrapper
########################################################################
#$ -pe orte 8
#$ -q i08m3
#$ -j y
#$ -M vdrouin@physics.rutgers.edu
#$ -m e
#$ -v WIEN_DMFT_ROOT,WIENROOT,LD_LIBRARY_PATH,PATH
########################################################################
# DON'T remove the following line!
source $TMPDIR/sge_init.sh
########################################################################
export SMPD_OPTION_NO_DYNAMIC_HOSTS=1
export OMP_NUM_THREADS=1
export PATH=.:$PATH
export MODULEPATH=/opt/apps/modulefiles:/opt/intel/modulefiles:/opt/pgi/modulefiles:/opt/gnu/modulefiles:/opt/sw/modulefiles

python3 ./mcpt-xy.py ${Lx}  ${Tmin} ${Tmax} ${NumberTemp} ${Type} ${NumCores} ${PreTherm} ${Therm} ${Measure}  >& log
python3 ./all_data_process.py ${Lx}  >& log2


