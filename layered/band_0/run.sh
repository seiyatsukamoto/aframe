#!/bin/bash
# Export environment variables
export AFRAME_TRAIN_DATA_DIR=/home/seiya.tsukamoto/aframe/layered/data/train
export AFRAME_TEST_DATA_DIR=/home/seiya.tsukamoto/aframe/layered/data/test
export AFRAME_TRAIN_RUN_DIR=/home/seiya.tsukamoto/aframe/layered/band_0/training
export AFRAME_CONDOR_DIR=/home/seiya.tsukamoto/aframe/layered/band_0/condor
export AFRAME_RESULTS_DIR=/home/seiya.tsukamoto/aframe/layered/band_0/results
export AFRAME_TMPDIR=/home/seiya.tsukamoto/aframe/layered/band_0/tmp/

LAW_CONFIG_FILE=/home/seiya.tsukamoto/aframe/layered/band_0/sandbox.cfg uv run --directory /home/seiya.tsukamoto/aframe law run aframe.pipelines.sandbox.Sandbox --workers 5 --gpus 0,1 --dev
