#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("openai/whisper-large-v3")
BATCH_SIZE=64

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=1 \
        --torch_compile \
        --compile_mode "reduce-overhead" \
        --max_eval_samples=-1 \
        --max_new_tokens=256

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=1 \
        --torch_compile \
        --compile_mode "reduce-overhead" \
        --max_eval_samples=-1 \
        --max_new_tokens=256
        
    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR
    mv "results" "short-form-results"

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/meanwhile" \
        --dataset="default" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --compile_mode "reduce-overhead" \
        --max_eval_samples=-1 \
        --max_new_tokens=256

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/tedlium-long-form" \
        --dataset="default" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --torch_compile \
        --compile_mode "reduce-overhead" \
        --max_eval_samples=-1 \
        --max_new_tokens=256

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
