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
unset $SERVER
unset $ONLINE_TRAINER
unset $ESTIMATOR
unset $OPENSHIFT_DEPLOY

DEPLOY_OPTIONS=$1
for opt in ${DEPLOY_OPTIONS}; do export $opt=true; done;

echo DEPLOY_OPTIONS=${DEPLOY_OPTIONS}

version=$(kubectl version| grep 'Client Version' | sed 's/.*v//g' | cut -b -4)
if [ 1 -eq "$(echo "${version} < 1.21" | bc)" ]
then
    echo "You need to update your kubectl version to 1.21+ to support kustomize"
    exit 1
fi

echo "Preparing manifests..."

if [ ! -z ${SERVER} ]; then
    echo "deploy model server"
    if [ ! -z ${ESTIMATOR} ]; then
        echo "add estimator-sidecar"
        # OPTS="ESTIMATOR SERVER" --> base
        cp ./manifests/base/estimate-with-server/kustomization.yaml ./manifests/base/kustomization.yaml
        if [ ! -z ${OPENSHIFT_DEPLOY} ]; then
            echo "patch openshift deployment for exporter (estimator-with-server)"
            # OPTS="ESTIMATOR SERVER OPENSHIFT_DEPLOY" --> base
            cp ./manifests/base/openshift/estimate-with-server/kustomization.yaml ./manifests/base/kustomization.yaml
        fi
    else
        # OPTS="SERVER" --> base
        cp ./manifests/base/serve-only/kustomization.yaml ./manifests/base/kustomization.yaml
        if [ ! -z ${OPENSHIFT_DEPLOY} ]; then
            echo "patch openshift deployment for exporter (serve-only)"
            # OPTS="SERVER OPENSHIFT_DEPLOY" --> base
            cp ./manifests/base/openshift/serve-only/kustomization.yaml ./manifests/base/kustomization.yaml
        fi
    fi
    
    if [ ! -z ${ONLINE_TRAINER} ]; then
        echo "add online trainer"
        # OPTS="... SERVER ONLINE_TRAINER" --> server
        cp ./manifests/server/online-train/kustomization.yaml ./manifests/server/kustomization.yaml
        if [ ! -z ${OPENSHIFT_DEPLOY} ]; then
            echo "patch openshift deployment for server (with online trainer)"
            # OPTS="... SERVER ONLINE_TRAINER OPENSHIFT_DEPLOY" --> server
            cp ./manifests/server/openshift/online-train/kustomization.yaml ./manifests/server/kustomization.yaml   
        fi
    else 
        # OPTS="... SERVER" --> server
        cp ./manifests/server/base/kustomization.yaml ./manifests/server/kustomization.yaml
        if [ ! -z ${OPENSHIFT_DEPLOY} ]; then
            echo "patch openshift deployment for server"
            # OPTS="... SERVER OPENSHIFT_DEPLOY" --> server
            cp ./manifests/server/openshift/serve-only/kustomization.yaml ./manifests/server/kustomization.yaml
        fi
    fi
elif [ ! -z ${ESTIMATOR} ]; then
    echo "add estimator-sidecar"
    # OPTS="ESTIMATOR" --> base
    cp ./manifests/base/estimate-only/kustomization.yaml ./manifests/base/kustomization.yaml
    if [ ! -z ${OPENSHIFT_DEPLOY} ]; then
        echo "patch openshift deployment for exporter (estimator-only)"
        # OPTS="ESTIMATOR OPENSHIFT_DEPLOY" --> base
        cp ./manifests/base/openshift/estimate-only/kustomization.yaml ./manifests/base/kustomization.yaml
    fi
fi

for opt in ${DEPLOY_OPTIONS}; do unset $opt; done; 

echo "Done $0"