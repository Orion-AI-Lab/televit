#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

max_epochs=50
experiment="iccv_hadr_televit_experiments"
batch_size=64
decoder=False
# patch size for local input
patch_size=16
# patch size for global input
global_patch_size=30
# Change debug to False if you want to run on the full dataset
debug=True

for target_shift in 1 2 4 8 16
do
  for use_indices in False True
  do
    for use_global_input in False True 
    do
      python src/train.py ++model.global_patch_size=${global_patch_size} ++model.use_global_input=${use_global_input} ++model.use_indices=${use_indices} datamodule.debug=${debug} datamodule.batch_size=${batch_size} target_shift=${target_shift} +model.vit_patch_size=${patch_size} ++model.sea_masked=False ++model.decoder=${decoder} trainer.max_epochs=${max_epochs} logger=wandb model.loss=ce model.encoder="vit" experiment=vit_global logger.wandb.project=${experiment}
    done
  done
done