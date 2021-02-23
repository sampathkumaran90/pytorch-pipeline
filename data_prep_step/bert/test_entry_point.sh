python3 \
    entry_point.py \
    --input_data gs://cloud-ml-nas-public/classification/imagenet/train* \
    --output_data gs://dpa26/imagenet/parquet/train2 \
    --region us-central1 \
    --staging_dir gs://dpa26/imagenet/temp/job2 
    