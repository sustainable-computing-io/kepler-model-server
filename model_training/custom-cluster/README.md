# local-dev-cluster

![GitHub](https://img.shields.io/github/license/sustainable-computing-io/local-dev-cluster)
[![units-test](https://github.com/sustainable-computing-io/local-dev-cluster/actions/workflows/test.yml/badge.svg)](https://github.com/sustainable-computing-io/local-dev-cluster/actions/workflows/test.yml)

This repo provides the scripts to create a local [kubernetes](kind/kind.sh)/[openshift](microshift/microshift.sh) cluster to be used for development or integration tests. It is also used in [Github action](https://github.com/sustainable-computing-io/kepler-action) for kepler.

## Prerequisites
- Locate your BCC lib and linux header.
- [`kubectl`](https://dl.k8s.io/release/v1.25.4)

## Start up
1. Modify kind [config](./kind/manifests/kind.yml) to make sure `extraMounts:` cover the linux header and BCC.
2. Export `CLUSTER_PROVIDER` env variable:
```
export CLUSTER_PROVIDER=kind/microshift
```
3. To setup local env run:
```
./main.sh up
```
4. To tear down local env run:
```
./main.sh down
```
## Container registry
There's a container registry available which is exposed at `localhost:5001`.

## For kepler contributor
To set up a local cluster for kepler development We need to make the cluster connected with a local container registry.

### Bump version step for this repo
1. Check kubectl version.
2. Check k8s cluster provider's version(as KIND).
3. Check prometheus operator version.

## How to contribute to this repo
### A new k8s cluster provider
You are free to ref kind to contribute a k8s cluster, but we will have a checklist as kepler feature.
1. Set up the k8s cluster.
2. The connection between the specific registry and cluster, as for local development usage. We hope to pull the development image to the registry instead of a public registry.
3. Able to get k8s cluster config, for the test case.
4. Mount local path for linux kenerl and ebpf(BCC) inside kepler pod.
