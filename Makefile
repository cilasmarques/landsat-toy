build-nvcc:
	/usr/local/cuda/bin/nvcc -g ./src/*.cu -o ./main -lcublas -lcutensor

build-cuda-core:
	/usr/local/cuda/bin/nvcc -g ./src/cudaCores.cu -o ./main1

build-cuda-tensor:
	/usr/local/cuda/bin/nvcc -g ./src/cudaTensor.cu -o ./main2 -lcutensor -lcublas 

build-cublas:
	/usr/local/cuda/bin/nvcc -g ./src/cublasTensor.cu -o ./main3 -lcutensor  -lcublas 

execute:
	./main