<img alt="Priberam logo" src="priberam-650x240.png" width="250" align="right" />

# Sentiment Analysis #

## Can be used: ## 
* to process a dataset and report best attained F1-score;
* as RESTful web service for on-demand sentiment analysis (dockerization is also available).

## The repository contains: ## 
* links to some pre-trained word embeddings;
* corpora (from SemEval-2017 Task 4 subtask A).

## Pre-trained word embeddings ## 
You can download one of the following word embeddings  (from "DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis"): 
- [datastories.twitter.200d.txt](https://mega.nz/#!W5BXBISB!Vu19nme_shT3RjVL4Pplu8PuyaRH5M5WaNwTYK4Rxes): 200 dimensional embeddings
- [datastories.twitter.300d.txt](https://mega.nz/#!u4hFAJpK!UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I): 300 dimensional embeddings

Place the file(s) in the `Embeddings` folder.


## Create a virtualenv ## 
Linux  | Windows
------------- | -------------
pip install virtualenv  | pip install virtualenv
virtualenv --python /usr/bin/python3.6 venv	  | virtualenv venv
source venv/bin/activate  | venv\Scripts\activate.bat
pip install -r requirements.txt  | pip install -r requirements.txt 
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl (1)* | pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl (1)*
pip install torchvision  | pip install torchvision
pip install torchtext==0.2.3  | pip install torchtext==0.2.3 

(1)* replace "cpu" in link if you plan to use GPU: "cu80" for CUDA 8, "cu90" for CUDA 9.0, "cu92" for CUDA 9.2, ...


### Train ## 
python sentiment_analysis.py --train_config_path="train_config.json"
### Run as a web service ## 
python sentiment_analysis.py --rest_config_path="REST_config.json"

### Config file arguments
* preprocessing_style : "english"(english wikipedia) or "twitter"(english tweets)


### Automation ## 
1. Edit the train config file to select the parameters of the models to be trained.
2. Run the script in the train mode.
The target REST web service config file will be updated with the trained models. 
3. Building a docker image afterwards (using provided Dockerfile) will create a running docker image with a REST web service, 
automatically configured with the trained models (model files and config file).