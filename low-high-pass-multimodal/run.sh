#!/bin/bash
# Export environment variables
export AFRAME_TRAIN_DATA_DIR=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/data/train
export AFRAME_TEST_DATA_DIR=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/data/test
export AFRAME_TRAIN_RUN_DIR=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/training
export AFRAME_CONDOR_DIR=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/condor
export AFRAME_RESULTS_DIR=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/results
export AFRAME_TMPDIR=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/tmp/

# launch pipeline; modify the gpus, workers etc. to suit your needs
# note that if you've made local code changes not in the containers
# you'll need to add the --dev flag!
LAW_CONFIG_FILE=/home/seiya.tsukamoto/aframe/low-high-pass-multimodal/sandbox.cfg uv run --directory /home/seiya.tsukamoto/aframe law run aframe.pipelines.sandbox.Sandbox --workers 5 --gpus 0,2,3 --dev
