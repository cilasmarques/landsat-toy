#!/bin/bash

current_dir=$(dirname -- "$0")
parent_dir=$(dirname -- "$current_dir")
cd -P -- "$parent_dir"

OUTPUT_DATA_PATH=./output

make build-cuda-core
make build-cuda-tensor
make build-cublas 

for i in $(seq -f "%02g" 1 60); do
    # Executa ./main e passa todos os argumentos para ele
    ./main1 "$@" >> $OUTPUT_DATA_PATH/output.csv 2> $OUTPUT_DATA_PATH/error.txt &
    PID=$!
    wait $PID

    ./main2 "$@" >> $OUTPUT_DATA_PATH/output.csv 2> $OUTPUT_DATA_PATH/error.txt &
    PID=$!
    wait $PID

    ./main3 "$@" >> $OUTPUT_DATA_PATH/output.csv 2> $OUTPUT_DATA_PATH/error.txt &
    PID=$!
    wait $PID
  sleep 1
done

exit 0