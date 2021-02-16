FROM gcr.io/ml-pipeline/ml-pipeline-gcp:1.3.0

RUN pip install tensorflow==2.3.*

COPY *.py /component/
COPY data_prep_beam/*.py /component/data_prep_beam/

WORKDIR /component

ENTRYPOINT /bin/bash

#RUN pip install apache-beam[gcp]==2.27.*


