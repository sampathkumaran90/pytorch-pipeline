import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["resnet", "bert"],
    required=True,
    help="Target model for compilation",
)
parser.add_argument(
    "--target",
    type=str,
    choices=["kfp", "mp"],
    required=True,
    help="Target platform for compilation",
)

args = parser.parse_args()

is_kfp = args.target == "kfp"

if is_kfp:
    print("Building for KFP backend")
else:
    print("Building for Managed Pipelines backend")

import kfp

if is_kfp:
    from kfp.components import load_component_from_file, load_component_from_url
    from kfp import dsl
    from kfp import compiler
    from kfp.aws import use_aws_secret
    from typing import NamedTuple
    from kfp.components import InputPath, OutputPath, create_component_from_func

else:
    from kfp.v2.components import load_component_from_file, load_component_from_url
    from kfp.v2 import dsl
    from kfp.v2 import compiler

data_prep_op = load_component_from_file(f"data_prep_step/{args.model}/component.yaml")
train_model_op = load_component_from_file(f"training_step/{args.model}/component.yaml")
# deploy_model_op = load_component_from_file("kfserving/component.yaml")
list_item_op = load_component_from_file("list_step/component.yaml")

# list_item_op = load_component_from_url(
#     "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/filesystem/list_items/component.yaml"
# )
tensorboard_op = load_component_from_file(f"visualization_step/{args.model}/component.yaml")

viz_web_op = load_component_from_file(f"visualize_html/bert/component.yaml")


#metrics_op = load_component_from_file("component_template.yaml")

@dsl.pipeline(name="pytorchcnn", output_directory="/tmp/output")
def train_imagenet_cnn_pytorch():

    if args.model == "bert":
        data_prep_task = data_prep_op(
            input_data="",
            dataset_url="https://kubeflow-dataset.s3.us-east-2.amazonaws.com/ag_news_csv.tar.gz",
        )

        train_model_task = (train_model_op(trainingdata = data_prep_task.outputs["output_data"],
            maxepochs = 2,
            numsamples = 150,
            batchsize = 16,
            numworkers = 2,
            learningrate = 0.001,
            accelerator = "",
            bucketname = "kubeflow-dataset",
            foldername = "bertViz")
            .set_cpu_limit('4').
            set_memory_limit('14Gi')
        ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

        # list_item_op_task = list_item_op(train_model_task.outputs["TensorboardLogs"])

        tensorboard_task = tensorboard_op(board_path='%s' % train_model_task.outputs['TensorboardS3Path']).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

        # list_item_op_task = list_item_op(tensorboard_task.outputs["outputs"])

        viz_web_task = viz_web_op(board_path='%s' % train_model_task.outputs['WebS3Path']).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

    if args.model == "resnet":
        data_prep_task = data_prep_op(input_data="")

        list_item_op_task = list_item_op(data_prep_task.outputs["output_data"])

        train_model_task = (
            train_model_op(
                trainingdata=data_prep_task.outputs["output_data"],
                maxepochs=1,
                gpus=0,
                trainbatchsize="None",
                valbatchsize="None",
                trainnumworkers=4,
                valnumworkers=4,
                learningrate=0.001,
                accelerator="None",
                bucketname="kubeflow-dataset",
                foldername="Cifar10Viz",
            )
            .set_cpu_limit("4")
            .set_memory_limit("14Gi")
        ).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))

        tensorboard_task = tensorboard_op(board_path='%s' % train_model_task.outputs['TensorboardS3Path']).apply(use_aws_secret('aws-secret', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'))


if is_kfp:
    compiler.Compiler().compile(
        pipeline_func=train_imagenet_cnn_pytorch,
        package_path="pytorch_dpa_demo_kfp.yaml",
    )
else:
    compiler.Compiler().compile(
        pipeline_func=train_imagenet_cnn_pytorch,
        pipeline_root=PIPELINE_ROOT,
        output_path="pytorch_dpa_demo.json",
    )


"""
Namespace(
    checkpoint_root='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2069872589/ModelCheckpoint', 
    tensorboard_root='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2069872589/TensorboardLogs', 
    train_glob='gs://managed-pipeline-test-bugbash/20210130/pipeline_root/pavel/c14ec128-18d4-4980-b9f3-e1c6f4babb51/pytorchcnn-dj5sg-2878573190/output_data')
"""
