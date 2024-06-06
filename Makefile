build-nvcc:
	nvcc -I ./include -g ./src/*.cu -o ./main -std=c++14 -ltiff

execute:
	./main