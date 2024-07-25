#!/usr/bin/env bash
#
# This file is part of the Kepler project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2023 The Kepler Contributors
#

# require: 
# - awscli
# - unzip

set -e

BUCKET_NAME="${BUCKET_NAME:-kepler-power-model}"
HOST_MNT_PATH="${HOST_MNT_PATH:-/mnt}"
MACHINE_SPEC_DIR="machine_spec"
MACHINE_ID=""

PIPELINE_NAME="${PIPELINE_NAME:-std_v0.7}"
POWER_SOURCE="${POWER_SOURCE:-rapl-sysfs}"
MODEL_TYPE="${MODEL_TYPE:-AbsPower}"
FEATURE_NAME="${FEATURE_NAME:-BPFOnly}"
TRAINER_NAME="${TRAINER_NAME:-XgboostFitTrainer_0}"
DOWNLOAD_PATH=""

check_data() {
    aws s3api list-objects --bucket kepler-power-model --query "Contents[?contains(Key, '/$MACHINE_ID/data')]" --output json|jq -r '.[].Key'
}

# load_data: try loading data
load_data() {
    mkdir -p $HOST_MNT_PATH/data
    mkdir -p $HOST_MNT_PATH/data/$MACHINE_SPEC_DIR
    echo "Loading data to $HOST_MNT_PATH/data"
    aws s3api get-object --bucket $BUCKET_NAME --key /$MACHINE_ID/data/idle.json $HOST_MNT_PATH/data/idle.json 2>&1 >/dev/null
    echo "Idle data loaded"
    aws s3api get-object --bucket $BUCKET_NAME --key /$MACHINE_ID/data/idle.json $HOST_MNT_PATH/data/kepler_query.json 2>&1 >/dev/null
    echo "Stress data loaded"
    aws s3api get-object --bucket $BUCKET_NAME --key /$MACHINE_ID/data/$MACHINE_SPEC_DIR/$MACHINE_ID.json $HOST_MNT_PATH/data/$MACHINE_SPEC_DIR/$MACHINE_ID.json 2>&1 >/dev/null
    echo "Spec loaded"
}

# load all model path
load_model_all() {
    keys=$(aws s3api list-objects --bucket kepler-power-model --query "Contents[?contains(Key, '/models/$PIPELINE_NAME')]" --output json|jq -r '.[].Key')
    for key in $keys; do
        MODEL_LOC=$(echo $key|cut -d'/' -f 3-15)
        DIR=$(echo "$MODEL_LOC" | rev | cut -d'/' -f2- | rev)
        mkdir -p $DOWNLOAD_PATH/$DIR
        echo GET $key
        aws s3api get-object --bucket $BUCKET_NAME --key $key $DOWNLOAD_PATH/$MODEL_LOC 2>&1 >/dev/null
	done
}

# load model weight file
load_model_weight() {
    # load model
    key=/models/$PIPELINE_NAME/$POWER_SOURCE/$MODEL_TYPE/$FEATURE_NAME/$TRAINER_NAME/weight.json
    echo GET $key
    aws s3api get-object --bucket $BUCKET_NAME --key $key $DOWNLOAD_PATH/${POWER_SOURCE}_${MODEL_TYPE}.json  2>&1 >/dev/null
    # load metadata
    key=/models/$PIPELINE_NAME/$POWER_SOURCE/$MODEL_TYPE/$FEATURE_NAME/$TRAINER_NAME/metadata.json 
    echo GET $key
    aws s3api get-object --bucket $BUCKET_NAME --key $key $DOWNLOAD_PATH/${POWER_SOURCE}_${MODEL_TYPE}_metadata.json 2>&1 >/dev/null
    cat $DOWNLOAD_PATH/${POWER_SOURCE}_${MODEL_TYPE}_metadata.json|jq
}

# load model zip file and unzip
load_model_zip() {
    key=/models/$PIPELINE_NAME/$POWER_SOURCE/$MODEL_TYPE/$FEATURE_NAME/$TRAINER_NAME.zip
    echo GET $key
    aws s3api get-object --bucket $BUCKET_NAME --key $key tmp.zip 2>&1 >/dev/null
    mkdir -p $DOWNLOAD_PATH/$POWER_SOURCE
    unzip -o -d $DOWNLOAD_PATH/$POWER_SOURCE/$MODEL_TYPE tmp.zip
    cat $DOWNLOAD_PATH/$POWER_SOURCE/$MODEL_TYPE/metadata.json|jq
}

# get the command line arguments and run the matching function
case "$1" in
    check_data)
        # get the machine id from the argument
        if [ -z "$2" ]; then
            echo "Machine ID is not set"
            exit 1
        fi
        MACHINE_ID=$2
        check_data
        ;;
    load_data)
        # get the machine id from the argument
        if [ -z "$2" ]; then
            echo "Machine ID is not set"
            exit 1
        fi
        MACHINE_ID=$2
        load_data
        ;;
    load_model)
        # get the load format
        if [ -z "$2" ]; then
            echo "Target format is not set {all|weight|zip}"
            exit 1
        fi
        # get the machine id from the argument
        if [ -z "$3" ]; then
            echo "Machine ID is not set"
            exit 1
        fi
        # get output path
        if [ -z "$4" ]; then
            echo "Download path is not set"
            exit 1
        fi
        DOWNLOAD_PATH=$3
        mkdir -p $DOWNLOAD_PATH
        load_model_$2
        ;;
    *)
        echo "Usage: $0 {check_data|load_data|load_model}"
        exit 1
esac