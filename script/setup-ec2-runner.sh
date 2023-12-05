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

set -o errexit
set -o pipefail
set -o nounset

# Define instance parameters
AMI_ID="${AMI_ID:-"ami-0e83be366243f524a}" # Ubuntu Server 22.04 LTS (HVM), SSD Volume Type, x86_64
INSTANCE_TYPE="${INSTANCE_TYPE:-"c6i.metal"}" # c is for compute, 6 is 6th geneneration, i is for Intel, metal is for bare metal
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-YOUR_SECURITY_GROUP_ID}"
GITHUB_TOKEN="${GITHUB_TOKEN:-YOUR_TOKEN}"
GITHUB_REPO="${GITHUB_REPO:-"https://github.com/sustainable-computing-io"}"
REGION="${REGION:-us-east-2}"          # Region to launch the spot instance
DEBUG="${DEBUG:-false}"                # Enable debug mode

debug() {
    [ "$DEBUG" == "true" ] && echo "DEBUG: $@" 1>&2
}

# fail if security group id is not set
if [ -z "$SECURITY_GROUP_ID" ]; then
    echo "SECURITY_GROUP_ID is not set"
    exit 1
fi

# fail if github token is not set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GITHUB_TOKEN is not set"
    exit 1
fi

# fail if aws cli is not found
if ! command -v aws &> /dev/null
then
    echo "aws cli could not be found"
    exit 1
fi

# GitHub Runner setup script
# based on https://github.com/organizations/sustainable-computing-io/settings/actions/runners/new?arch=x64&os=linux
# github_token is set in the environment variable
USER_DATA=$(cat <<EOF
#!/bin/bash
sudo apt-get update
sudo apt-get install -y curl jq
# Create a folder
mkdir actions-runner && cd actions-runner
# Download the latest runner package
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
# Optional: Validate the hash
# echo "29fc8cf2dab4c195bb147384e7e2c94cfd4d4022c793b346a6175435265aa278  actions-runner-linux-x64-2.311.0.tar.gz" | shasum -a 256 -c
# Extract the installer
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
# Create the runner and start the configuration experience
./config.sh --url ${GITHUB_REPO} --token ${GITHUB_TOKEN}
# Last step, run it!
./run.sh
EOF
)

# Encode user data so it can be passed as an argument to the AWS CLI
ENCODED_USER_DATA=$(echo "$USER_DATA" | base64)

# we use spot instance, the bid price is determined by the spot price history, simply use the last price for now
# Fetching spot price history
debug "Fetching spot price history..."
BID_PRICE=$(aws ec2 describe-spot-price-history --instance-types $INSTANCE_TYPE \
    --product-descriptions "Linux/UNIX" --availability-zone ${REGION} \
    --query 'SpotPriceHistory[0].SpotPrice' --output text)

# Creating a spot instance request
debug "Creating a spot instance with an initial bid of ${BID_PRICE}"
# try 3 times, each time increase the bid price by 10%
for i in {1..3}
do
    INSTANCE_JSON=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE \
    --security-group-ids $SECURITY_GROUP_ID \
    --instance-market-options '{"MarketType":"spot", "SpotOptions": {"MaxPrice": ${BID_PRICE}}}' \
    --user-data $ENCODED_USER_DATA)

    # Extract instance ID
    INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.Instances[0].InstanceId')

    # Check if instance creation failed
    if [ -z "$INSTANCE_ID" ]; then
        echo "Failed to create instance with bid price ${BID_PRICE}"
        # get the latest bid price and increase the bid price by 10%
        BID_PRICE=$(aws ec2 describe-spot-price-history --instance-types $INSTANCE_TYPE \
            --product-descriptions "Linux/UNIX" --availability-zone ${REGION} \
            --query 'SpotPriceHistory[0].SpotPrice' --output text)
        BID_PRICE=$(echo "$BID_PRICE * 1.1" | bc)
        debug "Creating a spot instance with a new bid of ${BID_PRICE}"
        continue
    fi
done

# if instance id is still empty, then we failed to create a spot instance
# create on-demand instance instead
if [ -z "$INSTANCE_ID" ]; then
    echo "Failed to create spot instance, creating on-demand instance instead"
    INSTANCE_JSON=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE \
        --security-group-ids $SECURITY_GROUP_ID \
        --user-data $ENCODED_USER_DATA)

    # Extract instance ID
    INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.Instances[0].InstanceId')

    # Check if instance creation failed
    if [ -z "$INSTANCE_ID" ]; then
        echo "Failed to create on-demand instance"
        exit 1
    fi
fi

# Wait for instance to become ready
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID

# Check if wait command succeeded
if [ $? -ne 0 ]; then
    echo "Instance failed to become ready. Terminating instance."
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID
    exit 1
fi

# Output the instance ID to github output
echo "::set-output name=instance-id::$INSTANCE_ID"
