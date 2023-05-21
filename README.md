# ROVIO Webcam Robot Model Training

This repository is dedicated to training a model to control the ROVIO webcam robot. The model predicts API commands to align the webcam view with a reference image.

## Usage

### Prerequisites

Before using this repository, ensure that you have the following prerequisites:

- Python 3.x installed
- Anaconda or Miniconda package manager installed (for environment setup)

## Setup

1. Clone this repository to your local machine.
2. Navigate to the repository directory.

#### Option 1: Create and activate the environment manually

3. Create a new Conda environment using the provided `environment.yml` file:
```shell
conda env create -f environment.yml
```

4. Activate the newly created Conda environment:
```shell
conda activate rovio-env
```
#### Option 2: Create and activate the environment using Anaconda Navigator

3. Open Anaconda Navigator.
4. Click on the "Import" button and select the `environment.yml` file from the repository directory.
5. Click on the "Create" button to create the environment.
6. Once the environment is created, click on the "Home" tab, select the newly created environment from the list, and click on the "Play" button to activate it.

### Define ROVIO connection settings

1. Create a file named `rovio_lib/rovio_connection.py`.
2. In `rovio_connection.py`, define the following values:

```python
ROVIO_IP = 'your_rovio_ip'
ROVIO_LOGIN = 'user_login'
ROVIO_PASS = 'login_pass'
```

Replace 'your_rovio_ip', 'user_login', and 'login_pass' with the actual values users need to provide.

### Generating Data
1. Run the data generation script, e.g., generate.py.
2. Replace the 'store_path' in generate.py with an appropriate path value where you want to store the generated data.
```python
store_path = 'path_to_store_data'
```
Replace 'path_to_store_data' with the actual path where you want to store the generated data.

Note: Make sure the specified path exists and is writable.

### Model Architecture
The model architecture in this project combines features from a pre-trained ResNet50 model with additional layers for command classification, logistic regression for speed estimation, and logistic regression for unit prediction.

The ResNet50 model serves as a feature extractor, capturing high-level features from input images. These features are then passed through a classification layer to predict the appropriate command for the ROVIO robot. The classification layer maps the extracted features to different commands such as "move forward," "turn left," "turn right," and so on.

In addition to the command classification, the model employs logistic regression models to estimate the desired speed and unit of movement for the robot. These regression models take the extracted features from ResNet50 as input and provide outputs for the desired speed and unit.

By combining the ResNet50 features with these additional layers, the model is able to make predictions for both the appropriate command and the corresponding speed and unit of movement for the ROVIO robot based on input images.

Training this model involves fine-tuning the pre-trained ResNet50 model on a dataset specific to the ROVIO robot task and optimizing the parameters of the logistic regression models. The trained model can then be used to control the ROVIO robot's webcam alignment based on input images and provide commands, speed, and unit predictions for accurate movement.

In controlling the ROVIO robot, accurately predicting the command takes precedence over speed and duration/angle. To emphasize this, the configuration file includes a weighting mechanism that prioritizes the command classification relative to the other components. By assigning a higher weight to the command classification, the model focuses on issuing correct commands for precise alignment with the reference image. This flexibility allows adjusting the model's behavior while considering speed and duration/angle as secondary factors in decision-making. The objective is to optimize command accuracy for accurate webcam alignment.

### Training the Model
1. Replace default_root_dir in `model/train_model.py` with desired checkpoint location
2. Run `model/train_model.py`
### Testing the Model
To evaluate model performance use `command_error_analysis.ipynb` Jupyter notebook.

### Inference
Inference example is in `inference_test.ipynb` Jupyter notebook.
### Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvement, please submit a pull request or open an issue.

### License
my_rovio distributed under the UB Public License (UBPL) version 1.0.