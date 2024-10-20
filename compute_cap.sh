#!/bin/bash

# Obtém o nome da GPU usando o nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)

# Mapeamento das GPUs para suas Compute Capabilities
case "$GPU_NAME" in
  *"RTX 4090"*) echo "89" ;;  # NVIDIA GeForce RTX 4090
  *"RTX 4080"*) echo "89" ;;  # NVIDIA GeForce RTX 4080
  *"RTX 4070"*) echo "89" ;;  # NVIDIA GeForce RTX 4070
  *"RTX 4060"*) echo "89" ;;  # NVIDIA GeForce RTX 4060
  *"RTX 3090"*) echo "86" ;;  # NVIDIA GeForce RTX 3090
  *"RTX 3080"*) echo "86" ;;  # NVIDIA GeForce RTX 3080
  *"RTX 3070"*) echo "86" ;;  # NVIDIA GeForce RTX 3070
  *"RTX 3060"*) echo "86" ;;  # NVIDIA GeForce RTX 3060
  *"A100"*) echo "80" ;;      # NVIDIA A100
  *"V100"*) echo "70" ;;      # NVIDIA V100
  *"T4"*) echo "75" ;;        # NVIDIA T4
  *"P100"*) echo "60" ;;      # NVIDIA P100
  *"A30"*) echo "80" ;;       # NVIDIA A30
  *"A40"*) echo "86" ;;       # NVIDIA A40
  *"A10"*) echo "86" ;;       # NVIDIA A10
  *"Titan V"*) echo "70" ;;   # NVIDIA Titan V
  *"Titan RTX"*) echo "75" ;; # NVIDIA Titan RTX
  *"GTX 1080"*) echo "61" ;;  # NVIDIA GTX 1080
  *"GTX 1070"*) echo "61" ;;  # NVIDIA GTX 1070
  *"GTX 1060"*) echo "61" ;;  # NVIDIA GTX 1060
  *"GTX 1050"*) echo "61" ;;  # NVIDIA GTX 1050
  *"GTX 1660"*) echo "75" ;;  # NVIDIA GTX 1660
  *"Quadro RTX 8000"*) echo "75" ;;  # NVIDIA Quadro RTX 8000
  *"Quadro RTX 6000"*) echo "75" ;;  # NVIDIA Quadro RTX 6000
  *"Quadro T2000"*) echo "75" ;;     # NVIDIA Quadro T2000
  *"Quadro P4000"*) echo "61" ;;     # NVIDIA Quadro P4000
  *"Quadro K6000"*) echo "35" ;;     # NVIDIA Quadro K6000
  *"Tesla K80"*) echo "37" ;;        # NVIDIA Tesla K80
  *"Tesla P4"*) echo "61" ;;         # NVIDIA Tesla P4
  *"Tesla P100"*) echo "60" ;;       # NVIDIA Tesla P100
  *"Tesla T4"*) echo "75" ;;         # NVIDIA Tesla T4
  *"Tesla V100"*) echo "70" ;;       # NVIDIA Tesla V100
  *)
    echo "Unknown compute capability for GPU: $GPU_NAME"
    exit 1 ;;
esac
