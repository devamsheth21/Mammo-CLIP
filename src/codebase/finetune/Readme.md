# Finetune Folder Overview

This folder contains scripts for fine-tuning models in the Mammo-CLIP project. Here is a brief overview of the scripts:

## Fine-tuning Scripts

1. `finetune-accelerate.py`: This script is used to fine-tune the *Multiview* model on an Image(multiview)-text dataset called `ft_uw_individual`. It utilizes gradient accumulation of 2 and mixed precision with FP16 for distributed training on 4 GPUs. The configuration file used is `configs/finetune-config.yaml`. To run the script, use the following command: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune-accelerate.py
```

2. `finetune.py`: Another script for fine-tuning models with different configurations and datasets. To run the script, use the following command: `python finetune.py`.

3. `inference.py`: This script is used to generate embeddings using a fine-tuned model on new, unseen data. It saves the embeddings in HDF5 format with proper distributed handling. To run the script, use the following command: 
``` 
CUDA_VISIBLE_DEVICES=5 accelerate launch inference.py --exp_num 1 --data_config finetune-config.yaml 
```

4. `eval/recallevalv2.py`: This script is used to evaluate the recall performance of the fine-tuned model on a the whole UW-Image-text dataset. It takes the model name, experiment number, and embedding file as input parameters. To run the script, use the following command: 
```
python eval/recallevalv2.py --modelname='multiview' --exp_num 2 --embeddingfile /path/to/embeddings_exp2.h5
```

5. `eval/class_eval.py`: This script is used to evaluate classification performance metrics for the specified class, such as "birads" or "density". To run the script, use the following command: 
```
python eval/class_eval.py --classname=birads
```

6. `vision-lp.py`: This script is used to perform linear probing on the fine-tuned model by adding a linear layer `(512, n_classes)` on top of the vision model for the specified class, such as "birads" or "density". To run the script, use the following command: 
```
python vision-lp.py --classname=birads
```


For more detailed information on each script's usage and parameters, please refer to the individual script files.

## Utility Scripts

This folder also contains several utility scripts:

- `model_utils.py`: Contains utility functions for model loading and manipulation.
- `mv_utils.py`: Contains utility functions specific to multiview models.
- `loss.py`: Implements the contrastive loss function used in training.

## Notebooks

The following notebooks are available:

- `test-multiview.ipynb`: Jupyter notebook for testing multiview models.
- `testcode.ipynb`: Jupyter notebook for running various test codes and experiments.

## Datasets
- `datasets/uw_dataset.py`: Contains the dataset class for the UW dataset. It has dataset classes like `ft_uw_individual` for Multiview dataset (with left and right laterality mammograms in one sample of batch). For linear probing, we have `ft_uw_linear_probe`, which processes "birads" and "density" for the linear probe labels.. 
- `preprocess/preprocess_data.py` : This script, modifies the existing jsons and csvs for UW data and prepares it to pass into dataloader in correct format, adding group_ids , accession numbers, paths for L and R mammograms and Density,Birads categories.
``` 
python preprocess/preprocess_data.py
```
## Embeddings

- Embeddings are stored in HDF5 format in the `embeddings/` directory.

## Checkpoints

- Model checkpoints are stored in the `checkpoints/` directory, organized by experiment number.

## Logs and Results

- Logs are stored in the `logs/` directory.
- Results are stored in the `results/` directory.

## Running the Project

To run the various scripts, use the provided commands in the respective sections above. Make sure to adjust the paths and parameters as needed for your specific setup.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
