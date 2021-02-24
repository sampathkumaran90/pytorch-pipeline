FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install image pandas sklearn tensorboard tensorboardX tensorboard-plugin-wit

COPY *.py /component/

WORKDIR /component

ENTRYPOINT /bin/bash

