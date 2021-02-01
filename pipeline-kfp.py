import kfp
from kfp.components import load_component_from_file
#import kfp.gcp as gcp

from kfp import dsl
from kfp import compiler


dummy_importer = kfp.components.load_component_from_text("""
name: Importer
inputs:
- {name: path, type: String, description: 'Path to be converted to artifact'}
outputs:
- {name: output_gcs_path, type: Dataset, description: 'Dataset URI'}
implementation:
  container:
    image: google/cloud-sdk:slim
    command:
    - sh
    - -c
    - |
      set -e -x
      echo "$0" > "$1"
    - {inputValue: path}
    - {outputUri: output_gcs_path}
""")



data_prep_op = load_component_from_file("data_prep_step/component.yaml")
train_model_op = load_component_from_file("training_step/component.yaml")

USER='pavel'
PIPELINE_ROOT = 'gs://managed-pipeline-test-bugbash/20210130/pipeline_root/{}'.format(USER)


@dsl.pipeline(
    name = "pytorchcnn",
    output_directory=PIPELINE_ROOT
)
def traing_imagenet_cnn_pytorch():
    

    training_data_path = "gs://cloud-ml-nas-public/classification/imagenet/train*"
    
    importer_task = dummy_importer(path = training_data_path)

    data_prep_task = data_prep_op(region = 'us-central1', input_data = importer_task.outputs["output_gcs_path"])

    #test_training_data_location = data_prep_task.outputs["output_data"]
    #test_training_data_location = "gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/186556260430/us-central1/pytorchcnn-20210131142111/PreProcessImageData/output_data"

    train_model_task = (train_model_op(data_prep_task.outputs["output_data"]).
        set_cpu_limit('4').
        set_memory_limit('14Gi').
        add_node_selector_constraint(
            'cloud.google.com/gke-accelerator',
            'nvidia-tesla-k80').
        set_gpu_limit(1)
    )


compiler.Compiler().compile(
    pipeline_func = traing_imagenet_cnn_pytorch,
    #pipeline_root = PIPELINE_ROOT,
    package_path="pytorch_dpa_demo_kfp.yaml",
)
