python3 \
      data_prep_beam.py \
      --runner DataflowRunner \
      --project managed-pipeline-test \
      --region us-central1 \
      --staging_location gs://dpa26/imagenet/temp/db4348e2587a11ebbaa842010a604402 \
      --temp_location gs://dpa26/imagenet/temp/db4348e2587a11ebbaa842010a604402 \
      --setup_file ./setup.py \
      --input gs://cloud-ml-nas-public/classification/imagenet/train* \
      --output gs://dpa26/imagenet/parquet/train \
      --save_main_session True
