build-serial:
	/usr/local/cuda/bin/nvcc -g ./src/serial.cu -o ./serial -ltiff -std=c++14

build-cuda-core:
	/usr/local/cuda/bin/nvcc -g ./src/cudaCores.cu -o ./cudaCores -ltiff -std=c++14

build-cutensor:
	/usr/local/cuda/bin/nvcc -g ./src/cudaTensor.cu -o ./cuTensor -lcutensor -lcublas -ltiff -std=c++14

build-cublas:
	/usr/local/cuda/bin/nvcc -g ./src/cublasTensor.cu -o ./cublas -lcutensor -lcublas -ltiff -std=c++14

execute:
	./main