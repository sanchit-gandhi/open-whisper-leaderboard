#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("openai/whisper-tiny.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-medium.en" "openai/whisper-large" "openai/whisper-large-v2")
BATCH_SIZE=64

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=1 \
        --max_eval_samples=-1
        
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
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/tedlium-long-form" \
        --dataset="default" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/earnings21" \
        --dataset="full" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/earnings22" \
        --dataset="full" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=5

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
