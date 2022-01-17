
## Review-based Topographic Organization of Latent Classes

This repository contains code for the paper "*Product Rating Prediction through Interpretable Latent Class Modeling of User Reviews*" (Serra, Tino, Xu & Yao).

For each product category, the code contains all the necessary steps to reproduce the experiments. The scripts are implemented with Python 3.7, and have been tested with Windows OS.

After data preparation, our software comprises three major parts:
1. EM training with topographic organization of latent classes.
2. Rating prediction part.
3. Evaluation of the results through both quantitative and qualitative experiments.


### Data sets
In the repository we uploaded a zipped folder (`preprocessed_data.zip`) containing all the necessary data for running the experiments. The folder is organized in subfolders divided per category. The raw reviews, instead, are publicly available [here](https://jmcauley.ucsd.edu/data/amazon/). Please note that we use the 5-core version of these data sets.

### Dependencies
All the dependencies are installed if `pip install -r requirements.txt` is run.

### Utils
The python file `utils.py` contains file paths and hyperparameters needed to run all the scripts. The list of data sets to evaluate can be changed in this file.

### Data preprocessing
Before training, some data preparation is needed to run the experiments. This includes creating and saving data structures that will be used during the EM step to speed up the computational time.

### EM Training
Once we prepared the data, we can run the EM algorithm. To train it, run the following command:
 - `python EM_training.py -K={} -L={}`

where K and L represent the number of user and product classes (default values are 25 and 16 respectively).

### Rating Prediction Part
First, run the following script.
 - `python data_preparation_rating.py -K={} -L={}`

After running the EM algorithm we have all the probability assignments of users and products to their respective latent classes. Given the imposed topological organization, we can think of these quantities as images where each pixel represents a latent class and the corresponding value is the latent class probability assignment. This script is used to create the correct input for the architecture. Below, an example of the input transformation. 

<img src="https://github.com/GiuseppeSerra93/TLCM/blob/main/images/fig1.png" height="300">
 
Now, we can run the architecture using the following command:
 - `python rating_prediction_CNN.py -epochs={} -bs={} -lr={} -gpu={} -K={} -L={} -runs={}`
     - `epochs`: number of epochs (default value 200)
     - `bs`: batch size (default value 256)
     - `lr`: learning rate (default value 0.05)
     - `gpu`: GPU device ID (integer)
     - `K`: number of user latent classes (default value 25)
     - `L`: number of product latent classes (default value 16)
     - `runs`: number of runs for each category (default value 5)

### Results evaluation
To visualize and evaluate the results using the output of our experiments, we provided a Jupyter notebook file `results_visualization.ipynb`. This file contains all the instructions to reproduce and to manually explore the results contained in the paper.
