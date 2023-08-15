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
# Copyright 2022 The Kepler Contributors
#

# set options
# for example: ./set.sh "ESTIMATOR SERVER"
DEPLOY_OPTIONS=$1
for opt in ${DEPLOY_OPTIONS}; do export $opt=true; done;

echo ${DEPLOY_OPTIONS}

version=$(kubectl version --short | grep 'Client Version' | sed 's/.*v//g' | cut -b -4)
if [ 1 -eq "$(echo "${version} < 1.21" | bc)" ]
then
    echo "You need to update your kubectl version to 1.21+ to support kustomize"
    exit 1
fi

echo "Preparing manifests..."

if [ ! -z ${ESTIMATOR} ]; then
    echo "add estimator-sidecar"
    cp ./manifests/base/estimate-only/kustomization.yaml ./manifests/base/kustomization.yaml
fi

if [ ! -z ${SERVER} ]; then
    echo "deploy model server"
    if [ ! -z ${ESTIMATOR} ]; then
        # with estimator, replace estimate-only with estimate-with-server
        cp ./manifests/base/estimate-with-server/kustomization.yaml ./manifests/base/kustomization.yaml
    else
        cp ./manifests/base/serve-only/kustomization.yaml ./manifests/base/kustomization.yaml
    fi
    # default
    cp ./manifests/server/base/kustomization.yaml ./manifests/server/kustomization.yaml
    if [ ! -z ${OPENSHIFT_DEPLOY} ]; then
        # replace with openshift serve-only
        echo "openshift deployment"
        cp ./manifests/server/openshift/serve-only/kustomization.yaml ./manifests/server/kustomization.yaml
    fi

    if [ ! -z ${ONLINE_TRAINER} ]; then
        # replace with online-train
        echo "add online trainer"
        if [ ! -z ${OPENSHIFT_DEPLOY} ]; then 
            cp ./manifests/server/openshift/online-train/kustomization.yaml ./manifests/server/kustomization.yaml
        else
            cp ./manifests/server/online-train/kustomization.yaml ./manifests/server/kustomization.yaml
        fi
    fi
fi

for opt in ${DEPLOY_OPTIONS}; do unset $opt; done; 

echo "Done $0"