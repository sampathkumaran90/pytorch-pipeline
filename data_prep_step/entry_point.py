# pytype: skip-file

from __future__ import absolute_import

import argparse
import os
from os import path
import logging

import data_prep_beam.data_prep_beam as dp_package

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='input data gcs path')
    parser.add_argument('--output_data',  type=str, help='output data gcs path')
    parser.add_argument('--region', type=str, help='region in GCP to run the Dataflow job')
    parser.add_argument('--staging_dir',  type=str, help='staging directory in GCS for temp files')

    args = parser.parse_args()

    print(args)

    fixed_output_dir = args.output_data + "/prefix"
    print("fixed output dir: {}".format(fixed_output_dir))

    beam_options = dp_package.ExtendedOptions.from_dictionary(
        {
            "runner": "DataflowRunner",
            "project": "managed-pipeline-test",
            "region" : args.region,
            "staging_location" : args.staging_dir,
            "temp_location" : args.staging_dir,
            "setup_file" : "./setup.py",
            "input" : args.input_data, 
            "output" : fixed_output_dir,
            "save_main_session" : True
        }
    )

    dp_package.run_pipeline_component(beam_options)


