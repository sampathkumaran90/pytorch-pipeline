# pytype: skip-file

from __future__ import absolute_import

import argparse
import os
from os import path
import logging

import data_prep_bert_pytorch as dp_package

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='input data path')
    parser.add_argument('--output_data',  type=str,
                        help='output data gcs path')
    parser.add_argument('--vocab_file', type=str,
                        help='name of the vocab file to be saved')
    parser.add_argument('--vocab_file_url',  type=str, help='Vocab file url')

    args = parser.parse_args()

    print(args)

    fixed_output_dir = args.output_data + "/prefix"
    print("fixed output dir: {}".format(fixed_output_dir))

    # pipeline_options = dp_package.ExtendedOptions(

    # )

    pipeline_options = {
        "input": args.input_data,
        "output": fixed_output_dir,
        "VOCAB_FILE": args.vocab_file,
        "VOCAB_FILE_URL": args.vocab_file_url,
        "save_main_session": True
    }

    dp_package.run_pipeline_component(pipeline_options)
