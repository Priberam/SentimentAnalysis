FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install torchtext==0.2.3
EXPOSE 7000
CMD python ./sentiment_analysis.py --rest_config_path="REST_config.json"