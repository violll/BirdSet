#!/bin/bash
 
# for test_set in HSN NBP NES PER SNE SSW UHH POW
for test_set in SSW UHH POW
do
    echo "Fine-tuning EfficientNet on $test_set dataset"
    python birdset/train.py experiment="birdset_neurips24/$test_set/DT/efficientnet"
done