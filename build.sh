#gcloud auth application-default login

python3 gen_image_timestamp.py > curr_time.txt

export images_tag=$(cat curr_time.txt)
echo ++++ Building component images with tag=$images_tag

for COMPONENT in training_step data_prep_step
do
    cd ./$COMPONENT

    full_image_name=gcr.io/managed-pipeline-test/pytorch-dpa/$COMPONENT:$images_tag

    echo IMAGE TO BUILD: $full_image_name

    #cp ../gcs_utils.py ./

    docker build -t $full_image_name .
    docker push $full_image_name

    sed -e "s|__IMAGE_NAME__|$full_image_name|g" component_template.yaml > component.yaml
    cat component.yaml 

    cd ..
done

pwd
echo
echo Running pipeline compilation
python3 pipeline.py --target mp
python3 pipeline.py --target kfp


#echo 
#echo Deploying to Managed Platform

#python3 deploy_to_managed.py
