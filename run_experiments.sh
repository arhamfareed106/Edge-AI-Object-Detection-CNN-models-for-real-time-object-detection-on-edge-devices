#!/bin/bash
# Automated Experiment Runner for Edge AI Object Detection
# Runs all 12 model-optimization combinations

set -e  # Exit on error

echo "=================================================="
echo "Edge AI Object Detection - Automated Experiment Runner"
echo "Running 12 experiments (3 models × 4 optimizations)"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MODELS=("mobilenet" "squeezenet" "yolo")
OPTIMIZATIONS=("baseline" "quantization" "pruning" "combined")
TOTAL_EXPERIMENTS=$(( ${#MODELS[@]} * ${#OPTIMIZATIONS[@]} ))
EXPERIMENT_NUM=0

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Error: Virtual environment 'venv' not found"
        echo "Run setup.sh first: bash setup.sh"
        exit 1
    fi
fi

# Start timer
START_TIME=$(date +%s)

# Run experiments
for model in "${MODELS[@]}"; do
    for opt in "${OPTIMIZATIONS[@]}"; do
        EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
        
        echo ""
        echo -e "${BLUE}==================================================${NC}"
        echo -e "${YELLOW}Experiment $EXPERIMENT_NUM/$TOTAL_EXPERIMENTS${NC}"
        echo -e "${GREEN}Model: $model | Optimization: $opt${NC}"
        echo -e "${BLUE}==================================================${NC}"
        
        # Run experiment
        python src/main.py \
            --model "$model" \
            --optimization "$opt" \
            --mode single \
            --save_results \
            --verbose
        
        # Check if experiment succeeded
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Experiment $EXPERIMENT_NUM complete${NC}"
        else
            echo -e "${YELLOW}⚠ Experiment $EXPERIMENT_NUM had issues (continuing...)${NC}"
        fi
        
        # Brief pause between experiments
        sleep 2
        
    done
done

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo ""
echo "=================================================="
echo -e "${GREEN}All Experiments Complete!${NC}"
echo "=================================================="
echo ""
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Total duration: ${DURATION_MIN}m ${DURATION_SEC}s"
echo ""
echo "Results saved to: output/csv/experiment_results.csv"
echo ""

# Generate visualizations
echo "Generating visualizations..."
python src/visualization.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Graphs saved to: output/graphs/"
    echo ""
    echo "Generated graphs:"
    ls -1 output/graphs/*.png 2>/dev/null | while read file; do
        echo "  - $file"
    done
else
    echo "⚠ Visualization generation failed"
fi

echo ""
echo "=================================================="
echo "Experiment Summary"
echo "=================================================="
echo ""

# Display results summary if CSV exists
if [ -f "output/csv/experiment_results.csv" ]; then
    echo "Results from experiment_results.csv:"
    echo ""
    python3 -c "
import pandas as pd
df = pd.read_csv('output/csv/experiment_results.csv')
print(df[['model', 'optimization', 'fps', 'latency_ms', 'map_score', 'model_size_mb']].to_string(index=False))
print()
print(f'Total experiments: {len(df)}')
print(f'Average FPS: {df[\"fps\"].mean():.2f}')
print(f'Average Latency: {df[\"latency_ms\"].mean():.2f} ms')
print(f'Average mAP: {df[\"map_score\"].mean():.3f}')
"
else
    echo "⚠ Results CSV not found"
fi

echo ""
echo "=================================================="
echo "All done! Check output/ directory for results."
echo "=================================================="
