#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main_reta.py   --config configs \
                                            --datasets I/A/V/R/S \
                                            --backbone ViT-B/16 \
                                            --data-root dataset/TTA_Data \