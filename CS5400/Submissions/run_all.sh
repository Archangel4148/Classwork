#!/bin/bash

# Check if folder path is given
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/folder"
    exit 1
fi

FOLDER="$1"
SCRIPT="$FOLDER/run.sh"

# Check if run.sh exists and is executable
if [ ! -x "$SCRIPT" ]; then
    echo "Error: '$SCRIPT' not found or not executable."
    exit 2
fi

# List of arguments to use
ARGS=("sample_inputs/input1.txt" "sample_inputs/input2.txt" "sample_inputs/input3.txt" "sample_inputs/input4.txt" "sample_inputs/input5.txt" "sample_inputs/input6.txt" "sample_inputs/input7.txt" "sample_inputs/input8.txt" "sample_inputs/input9.txt" "sample_inputs/input_100.txt")

# Run the script with each argument
for ARG in "${ARGS[@]}"; do
    "$SCRIPT" "$ARG" output.txt
done
