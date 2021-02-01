import os
from os import path
import glob
from google.cloud import storage
from urllib.parse import urlparse

def parse_gcs_full_path(gcs_full_path):
    url_obj = urlparse(gcs_full_path, allow_fragments=False)
    bucket_name = url_obj.netloc
    gcs_path = url_obj.path[1:]
    print("Detected bucket {} and path '{}'".format(bucket_name, gcs_path))

    if bucket_name.strip() == "" or len(bucket_name.strip()) < 2:
        raise Exception("Bucket name is empty or too small '{}'".format(bucket_name))

    return (bucket_name, gcs_path)
    

def copy_local_file_to_gcs(local_file_path, gsc_file_path):
    bucket_name, gcs_path = parse_gcs_full_path(gsc_file_path)
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file_path)


def upload_string_to_gcs(data_string, gsc_file_path):
    bucket_name, gcs_path = parse_gcs_full_path(gsc_file_path)
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(data_string)
    

def copy_local_directory_to_gcs_bucket(local_path, bucket, gcs_path):
    all_local_files = [y for x in os.walk(local_path) for y in glob.glob(os.path.join(x[0], '*'))]
    for local_file in all_local_files:
        if not os.path.isfile(local_file):
            print("{} is a directory".format(local_file))
            continue

        print("Found {} of size {}".format(local_file, os.path.getsize(local_file)))
        remote_path = os.path.join(gcs_path, os.path.relpath(local_file, local_path))
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)
        print("Uploaded to {}".format(remote_path))

def copy_local_directory_to_gcs(local_path, gcs_full_path):
    print("Copying from {} to {}".format(local_path, gcs_full_path))
    
    url_obj = urlparse(gcs_full_path, allow_fragments=False)
    bucket_name = url_obj.netloc
    gcs_path = url_obj.path[1:]
    print("Detected bucket {} and path '{}'".format(bucket_name, gcs_path))
    
    if bucket_name.strip() == "" or len(bucket_name.strip()) < 2:
        raise Exception("Bucket name is empty or too small '{}'".format(bucket_name))

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    copy_local_directory_to_gcs_bucket(local_path, bucket, gcs_path)

def copy_gcs_to_local_directory(gcs_full_path, local_path):
    print("Copying from {} to {}".format(gcs_full_path, local_path))
    
    url_obj = urlparse(gcs_full_path, allow_fragments=False)
    bucket_name = url_obj.netloc
    gcs_path = url_obj.path[1:]
    print("Detected bucket {} and path '{}'".format(bucket_name, gcs_path))
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)
    for blob in blobs:
        print("Downloading {}".format(blob.name))

        dirs_count = blob.name.count('/')
        filename = blob.name.split('/')[dirs_count]
        if filename == "":
            continue

        blob.download_to_filename(os.path.join(local_path, filename))
