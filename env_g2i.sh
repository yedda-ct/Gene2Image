#!/usr/bin/env bash


export PROJECT_DIR=/data/horse/ws/chwu350f-g2i/Gene2Image/code
export VENV_DIR=/data/horse/ws/chwu350f-g2i/venv_piptorch
export RELEASE_MODULE=release/24.10
export GCCCORE_MODULE=GCCcore/13.2.0
export PYTHON_MODULE=Python/3.11.5
export DATA_DIR=$PROJECT_DIR/data/processed_data
export OUTPUT_DIR=$PROJECT_DIR/results
export CHECKPOINT_DIR=$OUTPUT_DIR


export COMMON="ALL,PROJECT_DIR,VENV_DIR,RELEASE_MODULE,GCCCORE_MODULE,PYTHON_MODULE,DATA_DIR,OUTPUT_DIR,CHECKPOINT_DIR"
