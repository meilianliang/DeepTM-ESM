#!/bin/bash

show_usage() {
    echo "Usage: $0 --input <CSV> --model <model.pkl> --output <output.csv> [--ogt|--no-ogt] [--mode <1|2|3>]"
    echo ""
    echo "Required arguments:"
    echo "  --input PATH       Path to input CSV file (must contain 'uniprot_id' and 'sequence' columns)"
    echo "  --model PATH       Path to trained model file (.pkl)"
    echo "  --output PATH      Path to output CSV file for predictions"
    echo ""
    echo "Optional arguments:"
    echo "  --ogt              Use OGT version (model trained with OGT, CSV must contain 'ogt' column)"
    echo "  --no-ogt           Use no-OGT version (model trained without OGT)"
    echo "  --mode NUM         Execution mode: 1 (skip ESM-2), 2 (predict only), 3 (all steps) [default: 3]"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Dataset detection:"
    echo "  If the model filename contains 'DeepSTABp' (case-insensitive), DATASET=DeepSTABp will be set automatically."
    echo "  For other datasets (Tm50, TmPred), no DATASET is set (default branch)."
    echo "  You can override by setting the DATASET environment variable manually before running this script."
    echo ""
    echo "Parameter files:"
    echo "  The following .npy files must exist in the same directory as the model file:"
    echo "    mean_noblhhm.npy, mean_ogt.npy, std_noblhhm.npy, std_ogt.npy"
    echo "  They will be copied to the project's Data/ directory before prediction."
}

MODE=3
USE_OGT=""
INPUT_FILE=""
MODEL_FILE=""
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --ogt)
            USE_OGT="yes"
            shift
            ;;
        --no-ogt)
            USE_OGT="no"
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_FILE" ] || [ -z "$MODEL_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: Missing required arguments."
    show_usage
    exit 1
fi

if [ -z "$USE_OGT" ]; then
    echo "Error: You must specify either --ogt or --no-ogt."
    show_usage
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file '$MODEL_FILE' does not exist."
    exit 1
fi

if [[ ! "$MODE" =~ ^[123]$ ]]; then
    echo "Error: Mode must be 1, 2, or 3."
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "Project root: $PROJECT_ROOT"

# 将模型路径转换为绝对路径
if [[ "$MODEL_FILE" != /* ]]; then
    MODEL_FILE="$(cd "$(dirname "$MODEL_FILE")"; pwd)/$(basename "$MODEL_FILE")"
fi
MODEL_DIR="$(dirname "$MODEL_FILE")"

NPY_FILES=("mean_noblhhm.npy" "mean_ogt.npy" "std_noblhhm.npy" "std_ogt.npy")
DATA_DIR="$PROJECT_ROOT/Data"

echo "Checking for parameter files in $MODEL_DIR ..."
for file in "${NPY_FILES[@]}"; do
    src="$MODEL_DIR/$file"
    if [ ! -f "$src" ]; then
        echo "Error: Required parameter file $file not found in $MODEL_DIR"
        exit 1
    fi
    cp "$src" "$DATA_DIR/"
    echo "Copied $file to $DATA_DIR/"
done

# ========== 修改后的环境变量检测逻辑 ==========
# 优先使用用户手动设置的 DATASET
if [ -n "$DATASET" ]; then
    FINAL_DATASET="$DATASET"
    echo "Using user-set DATASET=$FINAL_DATASET"
else
    # 否则根据模型文件名自动检测（仅 DeepSTABp 需要特殊处理）
    MODEL_BASENAME="$(basename "$MODEL_FILE")"
    if [[ "$MODEL_BASENAME" =~ [Dd][Ee][Ee][Pp][Ss][Tt][Aa][Bb][Pp] ]]; then
        FINAL_DATASET="DeepSTABp"
        echo "Detected DeepSTABp model. Will set DATASET=DeepSTABp for each command."
    else
        FINAL_DATASET=""
        echo "Using default dataset (no DATASET will be set)."
    fi
fi
# =============================================

if [ "$USE_OGT" == "yes" ]; then
    PREDICT_DIR="script"
    echo "Using OGT version (predictor in $PREDICT_DIR/)"
else
    PREDICT_DIR="script_noogt"
    echo "Using no-OGT version (predictor in $PREDICT_DIR/)"
fi
PREDICT_SCRIPT="$PROJECT_ROOT/$PREDICT_DIR/predict.py"

if [ ! -f "$PREDICT_SCRIPT" ]; then
    echo "Error: Predict script not found at $PREDICT_SCRIPT"
    exit 1
fi

GEN_PT="$PROJECT_ROOT/script_features/gen_pt.py"
GET_FEATURES="$PROJECT_ROOT/script_features/get_features_test.py"
GCM="$PROJECT_ROOT/script_features/gcm.py"

if [ "$MODE" -eq 3 ] && [ ! -f "$GEN_PT" ]; then
    echo "Error: gen_pt.py not found at $GEN_PT"
    exit 1
fi
if [ "$MODE" -ne 2 ] && [ ! -f "$GET_FEATURES" ]; then
    echo "Error: get_features_test.py not found at $GET_FEATURES"
    exit 1
fi
if [ "$MODE" -ne 2 ] && [ ! -f "$GCM" ]; then
    echo "Error: gcm.py not found at $GCM"
    exit 1
fi

if [[ "$INPUT_FILE" != /* ]]; then
    INPUT_FILE="$(cd "$(dirname "$INPUT_FILE")"; pwd)/$(basename "$INPUT_FILE")"
fi
if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_FILE")"; pwd)"
    OUTPUT_FILE="$OUTPUT_DIR/$(basename "$OUTPUT_FILE")"
fi

echo "Input file: $INPUT_FILE"
echo "Model file: $MODEL_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Mode: $MODE"
echo "========================================="

run_python() {
    local script="$1"
    local desc="$2"
    shift 2
    echo "Running: $desc"
    local script_dir="$(dirname "$script")"
    local script_name="$(basename "$script")"
    cd "$script_dir" || { echo "Error: Cannot enter directory $script_dir"; exit 1; }
    if [ -n "$FINAL_DATASET" ]; then
        DATASET="$FINAL_DATASET" python "$script_name" --input "$INPUT_FILE" "$@"
    else
        python "$script_name" --input "$INPUT_FILE" "$@"
    fi
    local ret=$?
    cd "$PROJECT_ROOT" || exit 1
    if [ $ret -ne 0 ]; then
        echo "Error: $desc failed."
        exit 1
    fi
    echo "Completed: $desc"
    echo "-----------------------------------------"
}

case $MODE in
    1)
        echo "Mode 1: Running steps 2, 3, and 4 only (skip ESM-2 embedding extraction)"
        run_python "$GET_FEATURES" "Step 2: Node feature creation (get_features_test.py)"
        run_python "$GCM" "Step 3: Edge feature creation (gcm.py)"
        run_python "$PREDICT_SCRIPT" "Step 4: Prediction (predict.py)" --model "$MODEL_FILE" --output "$OUTPUT_FILE"
        ;;
    2)
        echo "Mode 2: Running step 4 only (prediction)"
        run_python "$PREDICT_SCRIPT" "Step 4: Prediction (predict.py)" --model "$MODEL_FILE" --output "$OUTPUT_FILE"
        ;;
    3)
        echo "Mode 3: Running all steps"
        run_python "$GEN_PT" "Step 1: ESM-2 embedding extraction (gen_pt.py)"
        run_python "$GET_FEATURES" "Step 2: Node feature creation (get_features_test.py)"
        run_python "$GCM" "Step 3: Edge feature creation (gcm.py)"
        run_python "$PREDICT_SCRIPT" "Step 4: Prediction (predict.py)" --model "$MODEL_FILE" --output "$OUTPUT_FILE"
        ;;
esac

echo "All selected steps completed successfully. Predictions saved to $OUTPUT_FILE"