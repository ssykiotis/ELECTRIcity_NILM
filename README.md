# ELECTRIcity

Pytorch implementation of ELECTRIcity: An efficient Transformer for Non-Intrusive Load Monitoring.

## Data

The csv datasets could be downloaded here: [REDD](http://redd.csail.mit.edu/), [UK-DALE](https://jack-kelly.com/data/) and [Refit] (https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)

For Refit, we used a slightly different folder structure. We have created .txt files with columns labels that are required during data processing. Please place the .csv files in the Data Folder for the code to work.

The folder structure in the data folder should be:
data/
    UK_Dale/
            House_1/
                    .
                    .
                    .
            House_2/
                    .
                    .
                    .
    REDD/
          House_1/
                  .
                  .
                  .
          House_2/
                  .
                  .
                  .
    Refit
        Data/
                House2.csv
                House3.csv
                House5.csv
                House16.csv
        Labels/
                House2.txt
                House3.txt
                House5.txt
                House16.txt

## Training

This repository provides the end-to-end pipeline to train a model using ELECTRIcity. 


The required packages to run the code can be found in electricity.yml. Model training and testing can be done by running the electricity.py python file. 

```bash
python electricity.py
```

First, config.py provides all the hyperparameters required in the pipeline. Then, the script creates a dataset parser for either UK_Dale, Refit or Redd, depending on the choice of the user in config.py (arugment dataset_code). Trainer.py contains all the functions necessary to perform model training and testing. 

After the model is trained and tested, the following results are exported in 'results/dataset_code/appliance_name/:
1)best_acc_model.pth contains the exported weights of the model
2)results.pkl contains several metrics that were recorded during training.
3)test_result.json contains the ground truth labels during testing, as well as the model prediction. 


## Performance

Our models are trained for 100 epochs for every appliance in each dataset, with the hyperparameters that can be found in config.py.

### UK_Dale

<img src=results_uk_dale.png width=1000>

### REDD

<img src=results_redd.png width=1000>

### Refit

<img src=results_refit.png width=1000>


## Citing 
Please cite the following paper if you use our methods in your research:
```
@inproceedings{yue2020bert4nilm,
  title={BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring},
  author={Yue, Zhenrui and Witzig, Camilo Requena and Jorde, Daniel and Jacobsen, Hans-Arno},
  booktitle={Proceedings of the 5th International Workshop on Non-Intrusive Load Monitoring},
  pages={89--93},
  year={2020}
}
```


## Acknowledgement

We used [BERT4NILM](https://github.com/Yueeeeeeee/BERT4NILM) by Yue et. al. as template for our code. We would like to thank the authors for their valuable work that has inspired ours!
