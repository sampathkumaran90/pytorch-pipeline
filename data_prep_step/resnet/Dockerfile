FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install torchvision image pandas webdataset sklearn itertools

COPY *.py /component/
COPY data_prep/*.py /component/data_prep/

WORKDIR /component

ENTRYPOINT /bin/bash

#RUN pip install apache-test[gcp]==2.27.*


