#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

max_epochs=50
experiment="iccv_hadr_unet_experiments"
batch_size=64
# Change debug to False if you want to run on the full dataset
debug=True

for target_shift in 1 2 4 8 16
do
  echo "Experiment with target_shift=${target_shift}"
  python src/train.py target_shift=${target_shift} datamodule.debug=${debug} trainer.max_epochs=${max_epochs} datamodule.batch_size=${batch_size} logger=wandb logger.wandb.project=${experiment} model.loss=ce model.encoder="efficientnet-b1" experiment=unet
done