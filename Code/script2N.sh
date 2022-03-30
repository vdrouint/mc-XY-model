#!/bin/bash
########################################################################
#               Grid Engine Job Submission Script
########################################################################
#$ -j  y
#$ -M  vdrouin@physics.rutgers.edu
#$ -m  e
#$ -v LD_LIBRARY_PATH,OMP_NUM_THREADS
########################################################################
# DON'T remove the following line!
source $TMPDIR/sge_init.sh
########################################################################
source ~/.bashrc

export OMP_NUM_THREADS=1

export SMPD_OPTION_NO_DYNAMIC_HOSTS=1

export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64/:/opt/intel/lib/intel64/:/opt/intel/composer_xe_2013_sp1.3.174/compiler/lib/intel64/

python ./mcptdoublel.py ${N} ${J2} ${Lambda} ${NumberTemp} ${NumCores} ${Tmin} ${Tmax}>& log
python ./all_data_process.py ${N} ${J2} ${Lambda} >& log2


