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

# This script creates a spot instance request and waits for it to become ready.
# It then outputs the instance ID of the created instance.
#
# The script requires the AWS CLI to be installed and configured.

set -o pipefail

# Define instance parameters
AMI_ID="${AMI_ID:-ami-0e83be366243f524a}" # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type, x86_64
INSTANCE_TYPE="${INSTANCE_TYPE:-t2.micro}" # c6i.metal: c is for compute, 6 is 6th geneneration, i is for Intel, metal is for bare metal
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-YOUR_SECURITY_GROUP_ID}"
GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_TOKEN}"
GITHUB_REPO="${GITHUB_REPO:-"sustainable-computing-io/kepler-model-server"}"
REGION="${REGION:-us-east-2}"          # Region to launch the spot instance
DEBUG="${DEBUG:-false}"                # Enable debug mode

INSTANCE_ID=""                         # ID of the created instance

[ "$DEBUG" == "true" ] && set -x

# get the organization name from the github repo
ORG_NAME=$(echo "$GITHUB_REPO" | cut -d'/' -f1)
# get the repo name from the github repo
REPO_NAME=$(echo "$GITHUB_REPO" | cut -d'/' -f2)
# github runner name
RUNNER_NAME="self-hosted-runner-"$(date +"%Y%m%d%H%M%S")

debug() {
    [ "$DEBUG" == "true" ] &&  echo "DEBUG: $@" 1>&2
}

get_github_runner_token () {
        # fail if github token is not set
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "GITHUB_TOKEN is not set"
        exit 1
    fi

    # create a github runner registration token using the github token
    # https://docs.github.com/en/actions/hosting-your-own-runners/adding-self-hosted-runners#adding-a-self-hosted-runner-to-a-repository-using-an-invitation-url
    # https://docs.github.com/en/rest/reference/actions#create-a-registration-token-for-an-organization

    # get the token
    RUNNER_TOKEN=$(curl -s -XPOST -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        https://api.github.com/repos/${ORG_NAME}/${REPO_NAME}/actions/runners/registration-token | jq -r '.token')

    debug "runner token: " $RUNNER_TOKEN
    # fail if then length of runner token is less than 5
    if [ ${#RUNNER_TOKEN} -lt 5 ]; then
        echo "Failed to get runner token"
        exit 1
    fi
}

prep () {
    # fail if security group id is not set
    if [ -z "$SECURITY_GROUP_ID" ]; then
        echo "SECURITY_GROUP_ID is not set"
        exit 1
    fi

    get_github_runner_token
}

# create the user data script
create_uesr_data () {
    # Encode user data so it can be passed as an argument to the AWS CLI
    # FIXME: it appears that the user data is not passed to the instance
    # ENCODED_USER_DATA=$(echo "$USER_DATA" | base64 | tr -d \\n)
    cat <<EOF > user_data.sh
#!/bin/bash
apt-get update
apt-get install -y curl jq
# Create a folder
mkdir /tmp/actions-runner && cd /tmp/actions-runner
# Download the latest runner package
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
# Optional: Validate the hash
# echo "29fc8cf2dab4c195bb147384e7e2c94cfd4d4022c793b346a6175435265aa278  actions-runner-linux-x64-2.311.0.tar.gz" | shasum -a 256 -c
# Extract the installer
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
# Create the runner and start the configuration experience
# there is a bug in the github instruction. The config script does not work with sudo unless we set RUNNER_ALLOW_RUNASROOT=true
export RUNNER_ALLOW_RUNASROOT=true
./config.sh --replace --unattended --name ${RUNNER_NAME} --url https://github.com/${GITHUB_REPO} --token ${RUNNER_TOKEN}
# Last step, run it!
./run.sh
EOF
}
    
get_bid_price () {
    BID_PRICE=$(aws ec2 describe-spot-price-history --instance-types $INSTANCE_TYPE \
        --product-descriptions "Linux/UNIX" --region ${REGION} \
        --query 'SpotPriceHistory[0].SpotPrice' --output text)
}

run_spot_instance () {
    INSTANCE_JSON=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE \
    --security-group-ids $SECURITY_GROUP_ID --region ${REGION}  --region ${REGION} \
    --instance-market-options '{"MarketType":"spot", "SpotOptions": {"MaxPrice": "'${BID_PRICE}'" }}' \
    --block-device-mappings '[{"DeviceName": "/dev/sda1","Ebs": { "VolumeSize": 200, "DeleteOnTermination": true } }]'\
    --user-data file://user_data.sh)
}

run_on_demand_instance() {
    INSTANCE_JSON=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE \
    --security-group-ids $SECURITY_GROUP_ID --region ${REGION} \
    --block-device-mappings '[{"DeviceName": "/dev/sda1","Ebs": { "VolumeSize": 200, "DeleteOnTermination": true } }]'\
    --user-data file://user_data.sh)

}

terminate_instance () {
    # Terminate instance   
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region ${REGION} 
}

create_runner () {
    # GitHub Runner setup script
    create_uesr_data

    # we use spot instance, the bid price is determined by the spot price history, simply use the last price for now
    # Fetching spot price history
    debug "Fetching spot price history..."
    get_bid_price
    # Creating a spot instance request
    debug "Creating a spot instance with an initial bid of ${BID_PRICE}"
    # try 2 times, each time increase the bid price by 10%
    for i in {1..2}
    do
        run_spot_instance
        # Extract instance ID
        INSTANCE_ID=$(echo -n "$INSTANCE_JSON" | jq -r '.Instances[0].InstanceId')

        # Check if instance creation failed
        if [ -z "$INSTANCE_ID" ]; then
            echo "Failed to create instance with bid price ${BID_PRICE}"
            # get the latest bid price and increase the bid price by 10%
            get_bid_price
            BID_PRICE=$(echo "$BID_PRICE * 1.1" | bc)
            debug "Creating a spot instance with a new bid of ${BID_PRICE}"
            continue
        else
            break
        fi
    done

    # if instance id is still empty, then we failed to create a spot instance
    # create on-demand instance instead
    if [ -z "$INSTANCE_ID" ]; then
        echo "Failed to create spot instance, creating on-demand instance instead"
        run_on_demand_instance

        # Extract instance ID
        INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.Instances[0].InstanceId')

        # Check if instance creation failed
        if [ -z "$INSTANCE_ID" ]; then
            echo "Failed to create on-demand instance"
            exit 1
        fi
    fi
    rm user_data.sh
    # Wait for instance to become ready
    aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID --region ${REGION} 

    # Check if wait command succeeded
    if [ $? -ne 0 ]; then
        echo "Instance failed to become ready. Terminating instance."
        terminate_instance
        exit 1
    fi

    # Output the instance ID to github output
    echo "instance_id=$INSTANCE_ID" >> $GITHUB_OUTPUT
    echo "runner_name=$RUNNER_NAME" >> $GITHUB_OUTPUT
}

list_runner () {
    # list all the runners
    # https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28
    curl -s -X GET -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/repos/${ORG_NAME}/${REPO_NAME}/actions/runners | jq -r '.runners[] | .name'
}

unregister_runner () {
    # unregister the runner from github
    # https://docs.github.com/en/rest/reference/actions#delete-a-self-hosted-runner-from-an-organization
    # cannot delete by runner name, need to get the runner id first
    RUNNERS=$(curl -s -X GET -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/repos/${ORG_NAME}/${REPO_NAME}/actions/runners)
    ID=$(echo $RUNNERS | jq -r '.runners[] | select(.name=="'$RUNNER_NAME'") | .id ')
    curl -L -X DELETE -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/repos/${ORG_NAME}/${REPO_NAME}/actions/runners/${ID}
}

# get the command line arguments and run the matching function
case "$1" in
    create)
        prep
        create_runner
        ;;
    terminate)
        # get the instance id from the argument
        if [ -z "$2" ]; then
            echo "Instance ID is not set"
            exit 1
        fi
        INSTANCE_ID=$2
        terminate_instance
        ;;
    unregister)
        # get the runner name from the argument
        if [ -z "$2" ]; then
            echo "Runner name is not set"
            exit 1
        fi
        RUNNER_NAME=$2
        unregister_runner
        ;;
    list)
        list_runner 
        ;;
    *)
        echo "Usage: $0 {create|terminate|unregister|list}"
        exit 1
esac