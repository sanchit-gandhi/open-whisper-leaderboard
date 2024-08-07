#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("large-v3")

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device="cuda:0" \
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
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/tedlium-long-form" \
        --dataset="default" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/earnings21" \
        --dataset="default" \
        --split="full" \
        --device="cuda:0" \
        --max_eval_samples=5

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="distil-whisper/earnings22" \
        --dataset="full" \
        --split="test" \
        --device="cuda:0" \
        --max_eval_samples=5

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
