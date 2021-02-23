FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install pytorch-lightning

COPY components/* /component/

RUN pip install -r component/requirements.txt

COPY *.py /component/



