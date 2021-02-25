#gcloud auth application-default login

#adding GPU to KFP cluster:
#1) add GPU node pool:
# gcloud container node-pools create gpunodes3 --accelerator type=nvidia-tesla-k80,count=1 --zone us-central1-a  --num-nodes 1 --machine-type n1-highmem-8
#2) add GPU driver installer deamonset:
# kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

python3 gen_image_timestamp.py > curr_time.txt

export images_tag=$(cat curr_time.txt)
echo ++++ Building component images with tag=$images_tag

MODEL=bert

for COMPONENT in data_prep_step training_step visualization_step visualize_html
do
    cd ./$COMPONENT/$MODEL

    full_image_name=jagadeeshj/$COMPONENT:$images_tag

    echo IMAGE TO BUILD: $full_image_name

    #cp ../gcs_utils.py ./

    docker build -t $full_image_name .
    docker push $full_image_name

    sed -e "s|__IMAGE_NAME__|$full_image_name|g" component_template.yaml > component.yaml
    cat component.yaml 

    cd ../..
done

pwd
echo
echo Running pipeline compilation
# python3 pipeline.py --target mp
python3 pipeline.py --target kfp --model bert


#echo 
#echo Deploying to Managed Platform

#python3 deploy_to_managed.py
