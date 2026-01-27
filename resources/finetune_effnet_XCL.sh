#!/bin/bash

model_path='/home/gil/comp0173/model.ckpt'
 
for test_set in HSN NBP NES PER SNE SSW UHH POW
do
    echo "Fine-tuning EfficientNet on $test_set dataset"
    python birdset/train.py experiment="birdset_neurips24/$test_set/DT/efficientnet"
    # python train.py "experiment=local/$test_set/efficientnet_inference_XCL" "module.network.model.local_checkpoint=$model_path" "module.network.model.pretrain_info.hf_name=$test_set" seed=2 "logger.wandb.group=inference_efficientnet_seed2_xcl_$test_set" "trainer.devices=[2]" "tags=['$test_set', 'efficientnet', 'multilabel', 'inference', 'XCL', 'proper-validation']" "ckpt_path=$model_path" "module.network.model.pretrain_info.hf_pretrain_name=XCL" 
done