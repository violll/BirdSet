#!/bin/bash

model_path='/home/gil/comp0173/model.ckpt'

echo "Fine-tuning EfficientNet on HSN dataset"
python train.py \
    "experiment=birdset_neurips24/HSN/DT/efficientnet" \
    "module.network.model.local_checkpoint=$model_path" \
    "module.network.model.pretrain_info.hf_pretrain_name=XCL" \
    "trainer.devices=[0]"
