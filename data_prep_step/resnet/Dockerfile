FROM gcr.io/ml-pipeline/ml-pipeline-gcp:1.3.0

RUN pip install torch torchvision image pandas

COPY *.py /component/
COPY data_prep/*.py /component/data_prep/

WORKDIR /component

ENTRYPOINT /bin/bash

#RUN pip install apache-test[gcp]==2.27.*


