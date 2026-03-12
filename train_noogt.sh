#!/bin/bash

# 显示使用说明
show_usage() {
    echo "Usage: $0 <CSV file path> [mode]"
    echo "Example: $0 /path/to/your/data.csv 3"
    echo ""
    echo "Modes:"
    echo "  1: Run steps 2, 3, and 4 only (skip ESM-2 embedding extraction)"
    echo "  2: Run step 4 only (training)"
    echo "  3: Run all steps (1, 2, 3, 4)"
    echo ""
    echo "Note: CSV file should be located in the project root directory or its subdirectories"
    echo "Default mode is 3 (run all steps)"
}

# Check if input file parameter is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Get input file path
INPUT_FILE="$1"

# Get mode parameter (default to 3 if not provided)
MODE="${2:-3}"

# Validate mode parameter
if [[ ! "$MODE" =~ ^[123]$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be 1, 2, or 3."
    show_usage
    exit 1
fi

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' does not exist!"
    exit 1
fi

# Get absolute path of input file
if [[ "$INPUT_FILE" != /* ]]; then
    # If it's a relative path, convert to absolute path
    INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")"; pwd)/$(basename "$INPUT_FILE")"
fi

echo "The training data file: $INPUT_FILE"
echo "Selected mode: $MODE"
echo "========================================="

# Get project root directory (assuming script runs from project root)
PROJECT_ROOT="$(pwd)"
echo "Project root directory: $PROJECT_ROOT"

# Function to run a step with error checking
run_step() {
    local step_name="$1"
    local step_dir="$2"
    local script_name="$3"
    local step_num="$4"
    
    echo "Step $step_num: $step_name ($script_name)..."
    cd "$PROJECT_ROOT/$step_dir" || { echo "Cannot enter $step_dir directory"; exit 1; }
    python "$script_name" --input "$INPUT_FILE"
    
    # Check if step succeeded
    if [ $? -ne 0 ]; then
        echo "Error: $step_name step failed!"
        exit 1
    fi
    
    echo "$step_name completed!"
    echo "========================================="
}

# 根据模式执行相应的步骤
case $MODE in
    1)
        # Mode 1: Only steps 2, 3, and 4
        echo "Mode 1: Running steps 2, 3, and 4 only (skip ESM-2 embedding extraction)"
        echo "========================================="
        
        # Step 2: Run node feature creation - create_node_esm.py
        run_step "Running node feature creation" "script_features" "get_features.py" "2"
        
        # Step 3: Run edge feature creation - gcm.py
        run_step "Running edge feature creation" "script_features" "gcm.py" "3"
        
        # Step 4: Run training - eval.py
        run_step "Running training" "script_noogt" "main_train_with_valid.py" "4"
        ;;
        
    2)
        # Mode 2: Only step 4
        echo "Mode 2: Running step 4 only (training)"
        echo "========================================="
        
        # Step 4: Run training - eval.py
        run_step "Running training" "script_noogt" "main_train_with_valid.py" "4"
        ;;
        
    3)
        # Mode 3: All steps (default)
        echo "Mode 3: Running all steps"
        echo "========================================="
        
        # Step 1: Run ESM-2 embedding extraction - extract_esm_embeddings.py
        run_step "Running ESM-2 embedding extraction" "script_features" "gen_pt.py" "1"
        
        # Step 2: Run node feature creation - create_node_esm.py
        run_step "Running node feature creation" "script_features" "get_features.py" "2"
        
        # Step 3: Run edge feature creation - gcm.py
        run_step "Running edge feature creation" "script_features" "gcm.py" "3"
        
        # Step 4: Run training - eval.py
        run_step "Running training" "script_noogt" "main_train_with_valid.py" "4"
        ;;
esac

echo "All selected steps completed! Processing finished."