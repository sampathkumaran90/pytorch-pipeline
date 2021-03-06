import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, choices=["kfp","mp"], required=True, help='Target platform for compilation')

args = parser.parse_args()

is_kfp = args.target == "kfp"

if is_kfp:
    print("Building for KFP backend")
else:
    print("Building for Managed Pipelines backend")

#import kfp.gcp as gcp
import kfp

if is_kfp:
    from kfp.components import load_component_from_file 
    from kfp import dsl
    from kfp import compiler
else:
    from kfp.v2.components import load_component_from_file
    from kfp.v2 import dsl
    from kfp.v2 import compiler

# load components (note the components are not platform specific, but the importers are)
data_prep_op = load_component_from_file("data_prep_step/component.yaml")
train_model_op = load_component_from_file("training_step/component.yaml")
deploy_model_op = load_component_from_file("deploy_model/component.yaml")

# globals
USER='pavel'
PIPELINE_ROOT = 'gs://managed-pipeline-test-bugbash/20210130/pipeline_root/{}'.format(USER)

@dsl.pipeline(
    name = "pytorchcnn",
    output_directory=PIPELINE_ROOT
)
def train_imagenet_cnn_pytorch(
    training_data_path = "gs://cloud-ml-nas-public/classification/imagenet/train*"
    ):
    
    data_prep_task = data_prep_op(region = 'us-central1', input_data = training_data_path)

    #temp_input = "gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2878573190/output_data/prefix"
    #data_prep_task.outputs["output_data"])

    train_model_task = (train_model_op(trainingdata=data_prep_task.outputs["output_data"]).
        set_cpu_limit('4').
        set_memory_limit('14Gi').
        add_node_selector_constraint(
            'cloud.google.com/gke-accelerator',
            'nvidia-tesla-k80').
        set_gpu_limit(1)
    )

    deploy_model_task = deploy_model_op(modelcheckpoint = train_model_task.outputs["ModelCheckpoint"])


if is_kfp:
    compiler.Compiler().compile(
        pipeline_func = train_imagenet_cnn_pytorch,
        #pipeline_root = PIPELINE_ROOT, this doesn't work for some reason
        package_path="pytorch_dpa_demo_kfp.yaml",
    )
else:
    compiler.Compiler().compile(
        pipeline_func = train_imagenet_cnn_pytorch,
        pipeline_root = PIPELINE_ROOT,
        output_path="pytorch_dpa_demo.json",
    )

    
'''
Namespace(
    checkpoint_root='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2069872589/ModelCheckpoint', 
    tensorboard_root='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2069872589/TensorboardLogs', 
    train_glob='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2878573190/output_data')
'''