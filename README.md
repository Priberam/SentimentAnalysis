# Sentiment Analysis

##Can be used:
* to process a dataset and report best attained F1-score;
* as RESTful web service for on-demand sentiment analysis.

##The repository contains:
* links to some pre-trained word embeddings;
* corpora (from SemEval-2017 Task 4 subtask A).

##You can download one of the following word embeddings  (from "DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis"):
- [datastories.twitter.200d.txt](https://mega.nz/#!W5BXBISB!Vu19nme_shT3RjVL4Pplu8PuyaRH5M5WaNwTYK4Rxes): 200 dimensional embeddings
- [datastories.twitter.300d.txt](https://mega.nz/#!u4hFAJpK!UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I): 300 dimensional embeddings
Place the file(s) in the `Embeddings` folder.


##Create a virtualenv
pip install virtualenv
virtualenv --python /usr/bin/python3.6 venv					 # Windows: (equivalent)
source venv/bin/activate                                     # windows: venv\Scripts\activate.bat
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl #windows: pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl 
-> replace "cpu" in link if you plan to use GPU: "cu80" for CUDA 8, "cu90" for CUDA 9.0, "cu92" for CUDA 9.2, ...
pip install torchvision
pip install torchtext==0.2.3

##Train 
python sentiment_analysis.py --train

##Run as a web service
python sentiment_analysis.py --rest --model=Model/best_model

###Request arguments:
corpus = "twitter" , "english" #(using the word statistics from Twitter or from english Wikipedia)