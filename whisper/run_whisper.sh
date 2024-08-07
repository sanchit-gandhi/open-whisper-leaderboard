#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH
conda activate openai-whisper

MODEL_IDs=("large-v3")
DEVICE_INDEX=0

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

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
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/tedlium-long-form" \
        --dataset="default" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/earnings21" \
        --dataset="default" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/earnings22" \
        --dataset="full" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --max_eval_samples=5

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
