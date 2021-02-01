import sys, argparse, logging

import apache_beam as beam
from apache_beam.io import tfrecordio, parquetio
from apache_beam.transforms import combiners
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

import pyarrow as pa
import tensorflow as tf

feature_description = {
    'image/height': tf.io.FixedLenFeature((1,), tf.int64, default_value=[0]),
    'image/width': tf.io.FixedLenFeature((1,), tf.int64, default_value=[0]),
    'image/channels': tf.io.FixedLenFeature((1,), tf.int64, default_value=[0]),
    'image/colorspace': tf.io.FixedLenFeature((1,), tf.string, default_value=[b'']),
    'image/class/label': tf.io.FixedLenFeature((1,), tf.int64, default_value=[0]),
    'image/class/text': tf.io.FixedLenFeature((1,), tf.string, default_value=[b'']),
    'image/format': tf.io.FixedLenFeature((1,), tf.string, default_value=[b'']),
    'image/filename': tf.io.FixedLenFeature((1,), tf.string, default_value=[b'']),
    'image/encoded': tf.io.FixedLenFeature((1,), tf.string, default_value=[b''])
}

parquet_schema = {
    'image/height': pa.int64(),
    'image/width': pa.int64(),
    'image/channels': pa.int64(),
    'image/colorspace': pa.string(),
    'image/class/label': pa.int64(),
    'image/class/text': pa.string(),
    'image/format': pa.string(),
    'image/filename': pa.string(),
    'image/encoded': pa.binary()
}

def reformat_row(row):
    import pyarrow as pa
    
    out_row = {}
    for key, val in row.items():
        out_type = parquet_schema[key]
        np_val = val.numpy()
        if pa.types.is_integer(out_type):
          new_val = np_val[0].item()
        elif pa.types.is_binary(out_type):
          new_val = np_val[0]
        elif pa.types.is_string(out_type):
          new_val = np_val[0].decode('utf-8')
        else:
          raise "Unexpected type" # TODO: add a proper error

        out_row[key] = new_val
    return out_row

def run_pipeline(beam_options):    
    import tensorflow as tf

    with beam.Pipeline(options=beam_options) as p:
      (
        p 
        | "Read files in" >> \
            tfrecordio.ReadFromTFRecord(beam_options.input)
        | "Parse TF Examples from file" >> \
            beam.Map(lambda row: tf.io.parse_example(
                row, 
                feature_description)
            )
        | "Replace TF tensors with native types" >> \
            beam.Map(reformat_row)
        | "Write to Parquet" >> \
            parquetio.WriteToParquet(
                beam_options.output, 
                pa.schema(parquet_schema),
                num_shards=128
            )
      )

class ExtendedOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
      parser.add_argument('--input',
                        dest='input',
                        help='Input glob listing TFRecord files')
      parser.add_argument('--output',
                        dest='output',
                        help='Output root name and path for Parquet files')

def PrintOptions(options):
    for a in options.get_all_options(drop_default=True).items():
        print(a)


def main():
    print("Running data prep job")
    pipeline_args = ExtendedOptions()

    PrintOptions(pipeline_args)

    run_pipeline(
        pipeline_args
    )

def run_pipeline_component(options):
    print("Running data prep job from container")
    
    logging.getLogger().setLevel(logging.INFO)
    
    PrintOptions(options)

    run_pipeline(
        options
    )

    
#if __name__ == '__main__':
#    logging.getLogger().setLevel(logging.INFO)
#    main()
