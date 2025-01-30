# Nuclear mass predictions based on convolutional neural networks
This repository contains code and data for predicting nuclear masses using convolutional neural networks (CNNs). The project involves data processing, model training, and evaluation of predictions. This project has been developed by [David Morales](https://www.linkedin.com/in/david-morales-361b41282/).

The `TFG-David-Morales-DeLeón.pdf` file contains a detailed explanation of this project.

The following image depicts the results obtained:

![Image](Results%20backup/example_result.png)


## Data

The `Data` directory contains various datasets:
- `WS4.txt`: Raw data file for WS4.
- `WS4_cleaned.csv`: Cleaned data file for WS4.
- `df2016_2020_nono.csv`: Merged data for AME2016 and AME2020 without hashtags.
- `df2016_2020_yesyes.csv`: Merged data for AME2016 and AME2020 with hashtags.
- `mass2016.txt`: Raw data file for AME2016.
- `mass2016_cleaned_without_#.csv`: Cleaned data file for AME2016 without hashtags.
- `mass2020.txt`: Raw data file for AME2020.
- `mass2020_cleaned_without_#.csv`: Cleaned data file for AME2020 without hashtags.

## Scripts

### Data Processing
- `data_processing.py`: Contains functions for processing raw data files, cleaning data, and merging datasets.

### Model Training and Evaluation
- `CNN-I3.py`: Script for training and evaluating the I3 model.
- `CNN-I4.py`: Script for training and evaluating the I4 model.
- `new_nuclei_testing.py`: Script for testing the models on new nuclei data (AME2020).

### Utilities
- `utils.py`: Contains utility functions for data processing, plotting, and model evaluation.
- `multiple_scripts.py`: Script to execute multiple model runnings sequentially.

### Models

The `models.py` file contains the definitions of the convolutional neural network models used in this project:
- `CNN_I3`: Model class for CNN-I3.
- `CNN_I4`: Model class for CNN-I4.

## Configuration

The `config.yaml` file contains configuration settings for the project, including model parameters, data paths, and other settings.

## Usage

### Installing requirements
In order to run the scripts of the repository, you should download the necessary packages by running:
```sh
pip install -r requirements.txt
```

### Data Processing
To process the raw data files and generate cleaned datasets, run the `data_processing.py` script:
```sh
python data_processing.py
```

### Model training
To train the CNN models, run the respective scripts:
```sh
python CNN-I3.py
python CNN-I4.py
```

### Model evaluation
To evaluate the models on new nuclei data, run the `new_nuclei_testing.py` script:
```sh
python new_nuclei_testing.py
```

### Results
The results of the model training and evaluation are stored in the `CNN-I3 results` and `CNN-I4 results` results directories. These directories contain plots and metrics for their performance.

## License
This project is licensed under the MIT License. See the LICENSE file for details.