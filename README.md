
# ISIC 2018 Disease Classification Challenge
The challenge consists of classifying a given skin lesion image into the corresponding 7 classes namely:
- Melanoma
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)
- Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
- Dermatofibroma
- Vascular lesion 

The challenge requires you to submit binary classification for each of the __7__ disease classes, indicating the diagnosis of each lesion image.
The metric used to evalute the performance of various approaches is __normalized multi-class accuracy metric__ (balanced across categories). 
The clinincal rational being giving specific information and treatment of a lesion based on the class it belongs to.

## METHOD
We use a data augmentation and bagging ensemble architecture (__DABEA__), that uses data augmentation   and bagging in combination to generate multiple output vectors per model and then applies a 1×1 convolution layer as a meta-learner for combining different model outputs.

The details for the same can be found here.

### Data Generation
First the images need to be converted to 299x299 to be processed by deep neural nets.
```sh
find . -name '*.jpg' -exec sh -c 'echo "{}"; convert "{}" -resize 299x299\! `basename "{}"`' \;
```
To convert the images into the .tfr format for training, run the following command.
```sh
python ./datasets/convert_skin_lesions_2018.py TRAIN ./data/metadata.csv ./data/images299 ./data/data.tfr
```
We split the data in a __90:10__ train-validation ratio

### Model Training
Inception-Resnet-v2 and Inception-v4 are trained for 20,000 epochs with __Imagenet__ weights as initalisation over the train split.
We use per-image pixel normalization to train the individual models.
(The weights can be downloaded from here)

```sh
python train_image_classifier.py
    --train_dir=./running/inception_resnet_norm
    --dataset_dir=./data/data.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=train
    --model_name=inception_resnet_v2
    --preprocessing_name=dermatologic
    --checkpoint_path=./running/inception_resnet_v2.ckpt
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits
    --save_interval_secs=1800
    --optimizer=adam
    --normalize_per_image=1
    --max_number_of_steps=20000
    --experiment_tag="Model: InceptionResnetv2 Train: Deploy; Normalization: mode 1, erases mean"
    --experiment_file=./running/inception_resnet_norm/experiment.meta

python train_image_classifier.py 
    --train_dir=./running/inception_norm 
    --dataset_dir=./data/data.tfr 
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=train
    --model_name=inception_v4
    --preprocessing_name=dermatologic
    --checkpoint_path=./running/inception_v4.ckpt
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits
    --save_interval_secs=1800 
    --optimizer=adam
    --normalize_per_image=1
    --max_number_of_steps=20000 
    --experiment_tag="Model: Inceptionv4 Train: Deploy; Normalization: mode 1, erases mean"
    --experiment_file=./running/inception_norm/experiment.meta
```

To predic the performance of individual model, run the following

```sh
python predict_image_classifier.py
    --dataset_dir ./data/data.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=test
    --model_name=inception_resnet_v2
    --preprocessing_name=dermatologic
    --normalize_per_image=1
    --checkpoint_path=./running/inception_resnet_norm/

python predict_image_classifier.py
    --dataset_dir ./data/data.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=test
    --model_name=inception_v4
    --preprocessing_name=dermatologic
    --normalize_per_image=1
    --checkpoint_path=./running/inception_norm/
```
### DABEA
Multiple prediction outputs(over the internal validation split) are first stored for each base model using the following script.

```sh
python predict_image_classifier_modified.py
    --alsologtostderr
    --checkpoint_path=./running/inception_resnet_norm/model.ckpt-20000
    --dataset_dir=./data/data.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=test
    --model_name=inception_resnet_v2
    --preprocessing_name=dermatologic
    --id_field_name=id
    --eval_replicas=10
    --pool_scores=none
    --output_file=./running/val.features/inception_resnet_norm.feats
    --normalize_per_image=1

python predict_image_classifier_modified.py
    --alsologtostderr
    --checkpoint_path=./running/inception_norm/model.ckpt-20000
    --dataset_dir=./data/data.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=test
    --model_name=inception_v4
    --preprocessing_name=dermatologic
    --id_field_name=id
    --eval_replicas=10
    --pool_scores=none
    --output_file=./running/val.features/inception_norm.feats
    --normalize_per_image=1
```
These predictions are then combined together using bagging to form a combination vector with two channel

```sh
python ./assemble_features_modified.py ALL_LOGITS ./running/val.features/ ./running/val.features/val.comb_100_2norm.feats ./data/data2018.txt
```
Similarly, Multiple predictions are made for the test data provided by the challenge. and are combined to form a combination vector.

```sh
python ./datasets/convert_skin_lesions_2018.py TEST ./data/test.csv ./data/test ./data/test.tfr

python predict_image_classifier_modified.py
    --alsologtostderr
    --checkpoint_path=./running/inception_resnet_norm/model.ckpt-20000
    --dataset_dir=./data/test.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=test
    --model_name=inception_resnet_v2
    --preprocessing_name=dermatologic
    --id_field_name=id
    --eval_replicas=10
    --pool_scores=none
    --output_file=./running/test.features/inception_resnet_norm.feats
    --normalize_per_image=1

python predict_image_classifier_modified.py
    --alsologtostderr
    --checkpoint_path=./running/inception_norm/model.ckpt-20000
    --dataset_dir=./data/test.tfr
    --dataset_name=skin_lesions
    --task_name=label
    --dataset_split_name=test
    --model_name=inception_v4
    --preprocessing_name=dermatologic
    --id_field_name=id
    --eval_replicas=10
    --pool_scores=none
    --output_file=./running/test.features/inception_norm.feats
    --normalize_per_image=1
    
python ./assemble_features_modified.py ALL_LOGITS ./running/test.features/ ./running/test.features/test.comb_100_2norm.feats ./data/test.csv
```
A __1x1 conv__ layer is meta-learnt to combine the two channel input combination vector.

```sh
python train_conv_layer.py
    --input_training ./running/val.features/val.comb_100_2norm.feats
    --model_path ./neural_ensemble/model_100e_100c_2norm.ckpt
```

The final prediciton over the test data is produced by using the 1x1 conv layer trained over the internal validation split and using average pooling to combine the outputs.

```sh
python predict_conv_layer.py
    --input_data ./running/test.features/test.comb_100_2norm.feats
    --input_model neural_ensemble/model_100e_100c_2norm.ckpt
    --output_predictions pred.txt
    --output_metrics log_100e_100c_2norm.txt
    --pool_by_id avg
```

