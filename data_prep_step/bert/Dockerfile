FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install image pandas sklearn transformers torchtext pyarrow

COPY *.py /component/
# COPY data_prep/*.py /component/data_prep/

WORKDIR /component

ENTRYPOINT /bin/bash

#RUN pip install apache-beam[gcp]==2.27.*


