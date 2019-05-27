# Neural Program Induction for KBQA Without Gold Programs or Query Annotations
This repository contains the implementation of the Stable Sparse Reward Programmer model proposed in **Neural Program Induction for KBQA Without Gold Programs or Query Annotations** and links to download associated datasets. 

# Datasets
Datasets on Complex Question answering on Knowledge Bases, used for evaluating SRP (Sparse Reward Programmer)
1. Complex Sequential Question Answering (https://amritasaha1812.github.io/CSQA/) Dataset
2. WebQuestionsSP (https://www.microsoft.com/en-us/download/details.aspx?id=52763) Dataset

# Experiments on CQA
* **Step 1:** Inside the {repo_root}, create a folder named *data*

* **Step 2:** To Download Preprocessed Dataset:
  * For experiments on full CQA download 
    1. *preprocessed_data_full.zip* (https://drive.google.com/file/d/1jkMk2ReeGd6x5wzU8SOHlAwsPkzIBLf2/view?usp=sharing)
  * For experiments on subset of CQA having 10k QA pairs per type download 
    1. *preprocessed_data_alltypes_10k.zip* (https://drive.google.com/file/d/1BHkGU_9fHXC0fTTrvxsQrA2TiDYBVmCt/view?usp=sharing)
    2. *preprocessed_data_alltypes_10k_noisy.zip* (https://drive.google.com/file/d/1q4qomyYrLNG_2JOUxBMBVsToDl0JsRXI/view?usp=sharing)
  * For experiments on subset of CQA having 100 QA pairs per type download 
    1. *preprocessed_data_alltypes_100.zip* (https://drive.google.com/file/d/1JnzvSL7QKVNORdOYBE0qhRq7shvg7NsE/view?usp=sharing)
    2. *preprocessed_data_alltypes_100_noisy.zip* (https://drive.google.com/file/d/10HqWeSQEeicRHsh2lEjkib5Cw75FJ_vH/view?usp=sharing)
* **Step 3:** Put the preprocessed data inside the *data* folder 

* **Step 4:** To Download Preprocessed Knowledge Base:
For experiments on full CQA or subset of CQA having 10K questions per type, download full preprocessed *wikidata.zip* (https://drive.google.com/file/d/1_whud6L-VmYsFMDSSjw6pW7oPzojPp01/view?usp=sharing)
For experiments on subset of CQA having 100 questions per type, download the preprocessed version of the corresponding subset of wikidata, i.e. *wikidata_100.zip* (https://drive.google.com/file/d/1yInB34aS7GyUuSd7F8nz7ATSrMr3LyQv/view?usp=sharing)
* **Step 5:** For running any of the pytorch scripts,  go inside {repo_root}/code/NPI/pytorch and install the dependencies by running `$pip install -r requirements.txt`

## Experiments on the gold entity, relation, type linking data
* **Step 7:** The experiments on gold data from the above two datasets using Stable Reward Programmer are hosted at *SRP* (https://github.com/CIPITR/CIPITR)

## Experiments on the noisy entity, relation, type linking data 
* **Step 14:** This section contains the necessary experimentation details pertaining to the paper.

* **Step 15:** To do so go inside *{repo_root}/code/NPI/pytorch/noisy_WikiData_CSQA* folder

* **Step 16:** Each of the experiments are configured with a parameter file (in the parameter folder). There are three question types (simple, logical, quanti_count) in the parameter folder and each of the variants can be run on either the smaller subset of the dataset (i.e. CQA with 100 QA pairs per question type) or on the bigger subset (CQA with 10K QA pairs per question type). For e.g. for running on the *simple* question type on CQA-100 subset, use the parameter file parameters_simple_small.jso and to run on the CQA-10K dataset use the parameter file parameters_simple_big.json (*small* is for 100 QA pair subset of CQA and *big* is for 10K QA pair subset of CQA)

* **Step 17:** Create a folder *model*

* **Step 18:** To run the SRP model on any of the question categories (simple/logical/quanti_count) run `python train_SRP.py <parameter_file> <time_stamp>` (time-stamp is the ID of the current experiment run). Example script to run is in *run_SRP.sh*. This script will start the training as well as dump the trained model in the model and also run validation. 

* **Step 19:** To run the SSRP model on any of the question categories (simple/logical/quanti_count) run `python train_SSRP.py <parameter_file> <time_stamp>` (time-stamp is the ID of the current experiment run). Example script to run is in *run_SSRP.sh*. This script will start the training as well as dump the trained model in the model and also run validation. 

* **Step 20:** To load the trained model and run test, run `python load.py <parameter_file> <time-stamp>` (use the same ID as used during training)

For *e.g.* to train and test the models on *simple* question type on 100-QA pair subset of CQA:
1. `cd {repo_root}/code/NPI/pytorch/noisy_WikiData_CSQA`
2. `python train_SRP.py  parameters/parameters_simple_small.json SRP_Jan_7` *#this will create a folder model/simple_SRP_Jan_7 to dump the trained model*
3. `python load.py parameters/parameters_simple_big.json SRP_Jan_7` *#this will run the trained model on the big test data, as mentioned in the parameter file*
4. `python train_SSRP.py parameters/parameters_simple_small.json SSRP_Jan_7` *#this will create a folder model/simple_SSRP_Jan_7 to dump the trained model*
5. `python load.py parameters/parameters_simple_big.json SSRP_Jan_7` *#this will run the trained model on the big test data, as mentioned in the parameter file*