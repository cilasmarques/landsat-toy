build-nvcc:
	nvcc -g ./src/*.cu -o ./main -lcublas -lcutensor

build-cuda-core:
	nvcc -g ./src/cudaCores.cu -o ./main 

build-cuda-tensor:
	nvcc -g ./src/cudaTensor.cu -o ./main -lcutensor -lcublas 

build-cublas:
	nvcc -g ./src/cublasTensor.cu -o ./main -lcutensor  -lcublas 

execute:
	./main