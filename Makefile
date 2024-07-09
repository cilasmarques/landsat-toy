build-serial:
	/usr/local/cuda/bin/nvcc -g ./src/serial.cu -o ./serial

build-cuda-core:
	/usr/local/cuda/bin/nvcc -g ./src/cudaCores.cu -o ./cudaCores

build-cutensor:
	/usr/local/cuda/bin/nvcc -g ./src/cudaTensor.cu -o ./cuTensor -lcutensor -lcublas 

build-cublas:
	/usr/local/cuda/bin/nvcc -g ./src/cublasTensor.cu -o ./cublas -lcutensor  -lcublas 

execute:
	./main