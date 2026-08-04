# BirdSet Adaptation for Arctic Seabirds
Fork of DBD-research-group/BirdSet

This repository was created for the second coursework of UCL's COMP0173 module, which required students to adapt a model to a different domain. This modified codebase includes a script to reproduce the paper's DT baseline results, as well as a data readying script which converts the novel dataset into a HuggingFace compatible dataset for use in the BirdSet codebase. Additional modifications to the codebase were made to allow for the export of test data metadata for additional analyses.  
## Installation
Clone the repo and install dependencies.
```bash
git clone https://github.com/violll/BirdSet.git
```
There are two environments used in this project: 
1. **birdset**: used for running BirdSet training/inference
```bash
conda create -n birdset python=3.10
conda activate birdset
pip install -e .
```
2. **datas1-prep**: used for preprocessing and building the domain adapted dataset (ArcticBirdSounds; labeled as DataS1 in the repo).
```bash
conda env create -f scripts/datas1/environment.yaml
```
## Usage
### Reproduce baselines
- Run the provided script: `reproduce_paper_results.py`
    - pass `--fail-fast` if you want to return after one dataset fails rather than continuing
    - pass `--datasets "..."` to specify which datasets you'd like to run
    - pass `--skip-xcl` to skip the XCL evaluation - note, this dataset is very large 
- Metrics will print to the console and are saved to the following location: `logs/train/runs/<DatasetName>/<timestamp>/finalmetrics.json`
```bash
conda activate birdset
python scripts/datas1/reproduce_paper_results.py
```
#### Notes
- Substantial storage space is required, even if XCL is skipped. See the BirdSet huggingface datasets page for more details
- I ran into issues using the PER dataset; setting the below in the experiment config resolved the issue
```yaml
datamodule:
  dataset:
    n_workers: 1
```
### Domain Transfer: DataS1
- Activate the environment from the file located in `scripts/datas1`
```bash
conda activate datas1-prep
```
#### Assemble the dataset
- pass `--stages ...` to indicate which stages you would like run. Omitting the argument runs all stages.
- `preflight` - determines whether required files are present. 
    - You will need to [download the ArcticBirdSounds dataset](https://osf.io/b9trx/overview). Download the `DataS1.zip` file (1.3GB) and extract it into `data/DataS1`. 
    - If you are downloading the training data directly from xeno-canto, you will need to download XCL's parquet file from [HuggingFace](https://huggingface.co/datasets/DBD-research-group/BirdSet/tree/data/XCL) (filename `XCL_metadata.parquet`). Place this file in `data/xcl/`
- `download_train` downloads the dataset
    - specify source with `--xcl-source ..`
        - `--xcl-source hf` downloads the entire XCL dataset from HuggingFace
        - `--xcl-source xc` downloads the relevant training files from the source website xeno-canto. Note that some training files may not be available as they may have been removed from the website. The other download method downloads a snapshot of the dataset so all data will be available 
- `build` creates the dataset's parquet files required for use with the BirdSet codebase
```bash
python scripts/datas1/data_prep.py
```
#### Domain Transfer: Training and Evaluation
Switch back to the birdset environment before training
```bash
conda activate birdset
python birdset/train.py experiment="comp0173/DataS1/DT/efficientnet.yaml"
```
- The experiment `.yaml` also sets the following parameter, such that some metadata of each clip in the test set are saved to `logs/train/runs/<DatasetName>/<timestamp>/test_predictions.csv`
```yaml
module: 
    save_predictions: True
```