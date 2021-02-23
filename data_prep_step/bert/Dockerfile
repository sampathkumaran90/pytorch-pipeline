FROM gcr.io/ml-pipeline/ml-pipeline-gcp:1.3.0

RUN pip install torch torchvision image pandas sklearn transformers torchtext

COPY *.py /component/
# COPY data_prep/*.py /component/data_prep/

WORKDIR /component

ENTRYPOINT /bin/bash

#RUN pip install apache-beam[gcp]==2.27.*


