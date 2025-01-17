# PARAMS=-O3 -DNPROC=${NPROC} -DBSIZE=${BSIZE} -D${POWER} -DR=${R} -D${DEBUG} -D${KDEBUG} -D${POWER_DEBUG} --default-stream per-thread -lnvidia-ml -Xcompiler -lpthread,-fopenmp
# ARCH=sm_75
all:
	nvcc -lcutensor -rdc=true -arch sm_75 src/main.cu -o bin/prog
