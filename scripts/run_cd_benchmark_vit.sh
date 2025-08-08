#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main_reta.py   --config configs \
                                            --datasets caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101 \
                                            --backbone ViT-B/16 \
                                            --data-root dataset/CLIP_Data \