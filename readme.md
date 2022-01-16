## Review-based Topographic Organization of Latent Classes

This repository contains code for the paper "*Product Rating Prediction through Interpretable Latent Class Modeling of User Reviews*" (Serra, Tino, Xu & Yao).

For each product category, the code contains all the necessary steps to reproduce the experiments. The scripts are implemented with Python 3.6, and have been tested with Windows OS.

After data preparation, our software comprises three major parts:
1. EM training with topographic organization of latent classes.
2. Rating prediction part.
3. Evaluation of the results through both quantitative and qualitative experiments.


### Data sets
In the repository we uploaded a zipped folder (`preprocessed_data.zip`) containing all the necessary data for running the experiments. The folder is organized in subfolders divided per category. The raw reviews, instead, are publicly available [here](https://jmcauley.ucsd.edu/data/amazon/). Please note that we use the 5-core version of these data sets.

### Dependencies
All the dependencies are installed if `pip install -r requirements.txt` is run. 

### Data preprocessing
Before training, some data preparation is needed to run the experiments. This includes, among others, creating and saving data structures that will be used during the EM step to speed up the computational time.

### EM Training
Once we prepared the data, we can run the EM algorithm. To train it, run the following command:
 - `python EM_training.py -K={} -L={}`

where K and L represent the number of user and product classes (default values are 25 and 16 respectively).

The python file `utils.py` contains file paths and hyperparameters needed to run all the scripts. The list of data sets to evaluate can be changed in this file.

**Input**
- `users_map.pkl`: dictionary of the form `{userID: index}`
- `products_map.pkl`: dictionary of the form `{productID: index}`
- `{}_train.pkl`: replace `{}` with either `users_ID`, `products_ID`, `words`, `ratings`. Lists of training data for users, products, ratings and reviews (i.e. biterm lists).
- `{}_test.pkl`: replace `{}` with either `users_ID`, `products_ID`, `words`, `ratings`. Lists of test data for users, products, ratings and reviews (i.e. biterm lists).
- `keywords_mat.pkl`: file containing the $`V \times D`$ vocabulary matrix, i.e. the vector representations of the words contained in the vocabulary (note that the vector representations are taken from the pretrained language model).
 
**Output**
- `beta.pkl`: the file stores the $`\beta`$ matrix. This matrix will be used for generating textual explanations for the considered nodes.
- `z_users.pkl`: the file stores $`\theta_{i, k}`$, i.e. the probabilities of user $i$ to belong to cluster $k$. For all users and user clusters.
- `z_prods.pkl`: the file stores $`\theta_{j, \ell}`$, i.e. the probabilities of product $j$ to belong to cluster $\ell$. For all products and product clusters.
- `mse_evaluation.csv`: the file contains the train and test mean squared error (MSE) values for each evaluated epoch.
- `nll_evaluation.csv`: the file contains the train and test negative log-likelihood (NLL) values for each evaluated epoch.

### Results evaluation
We can use $`\beta`$, $`\theta_{i, k}`$ and $`\theta_{j, \ell}`$ to generate textual explanations for nodes. For more details about the generation of the explanations, we refer the readers to the paper. To visualize and evaluate the results using the output of our model, we provided a Jupyter notebook file `results_visualization.ipynb`. This file contains all the instructions to reproduce the images contained in the paper, and to manually explore the results.