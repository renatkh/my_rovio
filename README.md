# ROVIO Webcam Robot Model Training

This repository is dedicated to training a model to control the ROVIO webcam robot. The model predicts API commands to align the webcam view with a reference image.

## Usage

### Prerequisites

Before using this repository, ensure that you have the following prerequisites:

- Python 3.x installed
- Anaconda or Miniconda package manager installed (for environment setup)

### Setup

1. Clone this repository to your local machine.
2. Navigate to the repository directory.

#### Option 1: Create and activate the environment manually

3. Create a new Conda environment using the provided `environment.yml` file:

   ```shell
   conda env create -f environment.yml

4. Create a file named `rovio_lib/rovio_connection.py`.
5. In `rovio_connection.py`, define the following values:

   ```python
   ROVIO_IP = 'your_rovio_ip'
   ROVIO_LOGIN = 'user_login'
   ROVIO_PASS = 'login_pass'
