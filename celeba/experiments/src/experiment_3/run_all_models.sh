#!/bin/bash

# Set the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to run a model
run_model() {
    local model_file=$1
    local model_name=$(basename "$model_file" .py)
    
    echo "================================================"
    echo "Running $model_name at $(date)"
    echo "================================================"
    
    # Run the model with non-blocking matplotlib
    python "$model_file"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ $model_name completed successfully"
    else
        echo "❌ $model_name failed"
    fi
    echo "================================================"
    echo ""
}

# Main execution
echo "Starting model runs at $(date)"
echo ""

# Run each model in sequence
run_model "$SCRIPT_DIR/sl_1x1_bottom_right.py"
run_model "$SCRIPT_DIR/sl_5x5_bottom_right.py"
run_model "$SCRIPT_DIR/sl_10x10_bottom_right.py"
run_model "$SCRIPT_DIR/sl_1x1_center.py"
run_model "$SCRIPT_DIR/sl_5x5_center.py"
run_model "$SCRIPT_DIR/sl_10x10_center.py"
run_model "$SCRIPT_DIR/sl_1x1_corners.py"
run_model "$SCRIPT_DIR/sl_5x5_corners.py"
run_model "$SCRIPT_DIR/sl_10x10_corners.py"

echo "All models completed at $(date)" 
