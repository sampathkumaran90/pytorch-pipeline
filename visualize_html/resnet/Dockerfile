FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install pandas matplotlib seaborn

COPY *.py /component/

WORKDIR /component

ENTRYPOINT /bin/bash

