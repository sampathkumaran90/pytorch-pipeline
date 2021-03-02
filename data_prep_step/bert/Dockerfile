FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install image pandas sklearn transformers torchtext pyarrow

COPY *.py /component/

WORKDIR /component

ENTRYPOINT /bin/bash


