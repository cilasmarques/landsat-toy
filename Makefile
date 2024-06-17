build-nvcc:
	nvcc -g ./src/*.cu -o ./main -lcublas -lcutensor

execute:
	./main