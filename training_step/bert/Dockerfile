FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install pytorch-lightning==1.1.8 pyarrow

COPY components/* /component/

RUN pip install -r component/requirements.txt

COPY *.py /component/
