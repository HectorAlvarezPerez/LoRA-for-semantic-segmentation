# GTA5 to Cityscapes Semantic Segmentation with SegFormer and LoRA

This repository contains code for training and fine-tuning a SegFormer model for semantic segmentation, specifically transferring knowledge from the synthetic GTA5 dataset to the real-world Cityscapes dataset.  The project explores fine-tuning techniques with and without LoRA (Low-Rank Adaptation) to improve performance and efficiency. The use of PNG images for the GTA5 dataset is also explored.

## Overview

This project aims to improve semantic segmentation in urban environments by pre-training a SegFormer model on the GTA5 dataset and then fine-tuning it on the Cityscapes dataset. LoRA is used to reduce the computational cost of fine-tuning. The repository includes data loading, preprocessing, model definition, training pipelines, and evaluation metrics.

The repository contains the following scripts:

*   **`gta_segformer_trainer.py`:** Trains a SegFormer model on the GTA5 dataset from scratch using the Hugging Face `guimCC/gta5-cityscapes-labeling` dataset. Includes data loading, preprocessing, model definition, and training loop. Augmentation is also implemented with `ColorJitter` and `RandomHorizontalFlip`.

*   **`gta_to_cityscapes_no_lora.py`:** Fine-tunes a pre-trained SegFormer model (trained on GTA5) on the Cityscapes dataset *without* LoRA. Loads the pre-trained model and fine-tunes it.

*   **`lora_gta_to_cityscapes.py`:** Fine-tunes a pre-trained SegFormer model on the Cityscapes dataset *using* LoRA. Uses the `peft` library to efficiently fine-tune the model.

*   **`gta_segformer_trainer_png.py`:** Trains a SegFormer model on the GTA5 dataset from scratch, but loads the data from `.png` images using paths specified in `all_gta_images.txt` and `all_gta_masks.txt`. This allows training from custom or pre-processed GTA5 data.

## Datasets

This project uses the following datasets:

*   **GTA5:** A synthetic dataset of urban scenes rendered from Grand Theft Auto V, used for pre-training. The `gta_segformer_trainer.py` script utilizes the `guimCC/gta5-cityscapes-labeling` dataset from the Hugging Face Hub.  The `gta_segformer_trainer_png.py` script expects image paths in  `./all_gta_images.txt` and `masks` in `./all_gta_masks.txt`.

*   **Cityscapes:** A real-world dataset of urban scenes, used for fine-tuning. The scripts expect a local copy of the Cityscapes dataset located at `"../cityscapes_train_1000_dataset_v3"`.

## Dependencies

Install the following Python packages:

