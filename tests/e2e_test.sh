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

# Get the directory of the currently executing script_
set -eu -o pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
declare -r PROJECT_ROOT

source "$PROJECT_ROOT/hack/utils.bash"

declare -r NS="kepler"
declare -r EXPORTER="kepler-exporter"
declare -r MODEL_SERVER="kepler-model-server"
declare -r ESTIMATOR="estimator"
declare -r SERVER_API="server-api"
declare -r DB="model-db"
declare -r TIMEOUT="5m"

declare ENABLE_ESTIMATOR=false
declare ENABLE_SERVER=false
declare ENABLE_TEST=false
declare ENABLE_DB=false
declare SHOW_HELP=false

declare KEPLER_CFG="kepler-cfm"
declare LOGS_DIR="tmp/e2e"

declare MODEL_CONFIG=$PROJECT_ROOT/manifests/test/patch-estimator-sidecar.yaml
declare POWER_REQUEST_CLIENT=$PROJECT_ROOT/manifests/test/power-request-client.yaml
declare MODEL_REQUEST_CLIENT=$PROJECT_ROOT/manifests/test/model-request-client.yaml
declare FILE_SERVER=$PROJECT_ROOT/manifests/test/file-server.yaml

get_component() {
	local component="$1"
	kubectl get pods -n "$NS" -l app.kubernetes.io/component="$component" -oname
}

get_kepler_log() {
	info "Getting Kepler logs"
	kubectl logs -n "$NS" "$(get_component exporter)" -c "$EXPORTER"
}

get_estimator_log() {
	info "Getting Estimator logs"
	kubectl logs -n "$NS" "$(get_component exporter)" -c "$ESTIMATOR"
}

get_server_log() {
	info "Getting Model Server logs"
	kubectl logs -n "$NS" "$(get_component model-server)" -c "$SERVER_API"
}

get_db_log() {
	info "Getting Model DB logs"
	kubectl logs -n "$NS" "$DB"
}

init_logs_dir() {
	rm -rf "$LOGS_DIR-prev"
	mv "$LOGS_DIR" "$LOGS_DIR-prev" || true
	mkdir -p "$LOGS_DIR"
}

must_gather() {
	header "Running must gather"
	kubectl logs -n "$NS" "$(get_component exporter)" -c "$EXPORTER" | tee "$LOGS_DIR"/exporter.log || true
	kubectl logs -n "$NS" "$(get_component exporter)" -c "$ESTIMATOR" | tee "$LOGS_DIR"/estimator.log || true
	kubectl logs -n "$NS" "$(get_component model-server)" -c "$SERVER_API" | tee "$LOGS_DIR"/server.log || true
	kubectl logs -n "$NS" "$DB" --all-containers | tee "$LOGS_DIR"/db-all.log || true
	kubectl describe -n "$NS" daemonset "$EXPORTER" | tee "$LOGS_DIR"/exporter-describe || true
	kubectl describe -n "$NS" deployment "$MODEL_SERVER" | tee "$LOGS_DIR"/server-describe || true
	kubectl describe -n "$NS" pod "$DB" | tee "$LOGS_DIR"/db-describe || true
	kubectl get service -n "$NS" "$EXPORTER" -o yaml | tee "$LOGS_DIR"/exporter-svc.yaml || true
	kubectl get service -n "$NS" "$MODEL_SERVER" -o yaml | tee "$LOGS_DIR"/server-svc.yaml || true
	kubectl get service -n "$NS" "$DB" -o yaml | tee "$LOGS_DIR"/db-svc.yaml || true
}

wait_for_kepler() {
	local ret=0
	kubectl rollout status ds "$EXPORTER" -n "$NS" --timeout $TIMEOUT || ret=1
	# kubectl describe ds -n "$NS" "$EXPORTER" || ret=1
	kubectl get pods -n "$NS" || ret=1
	return $ret
}

wait_for_server() {
	local ret=0
	kubectl rollout status deploy "$MODEL_SERVER" -n "$NS" --timeout $TIMEOUT || ret=1
	wait_for_keyword 10 10 server "initial pipeline is loaded" "server cannot load initial pipeline" || ret=1
	wait_for_keyword 10 10 server "Press CTRL\+C to quit" "server has not started yet" || ret=1
	return $ret
}

wait_for_db() {
	local ret=0
	kubectl wait -n "$NS" --for=jsonpath='{.status.phase}'=Running pod/"$DB" --timeout $TIMEOUT || ret=1
	wait_for_keyword 10 10 db "Http File Serve Serving" "model-db is not serving" || ret=1
	return $ret
}

wait_for_keyword() {
	local max_tries="$1"
	local delay="$2"
	local component="$3"
	local keyword="$4"
	local msg="$5"
	shift 5

	local -i tries=0
	local -i ret=1

	while [[ $tries -lt $max_tries ]]; do
		[[ $(get_"$component"_log) =~ $keyword ]] && {
			ret=0
			break
		}
		tries=$((tries + 1))
		echo "  ... [$tries / $max_tries] waiting ($delay secs) - $msg" >&2
		sleep "$delay"
	done

	return $ret
}

check_estimator_set_and_init() {
	wait_for_keyword 10 10 kepler "Model Config NODE_COMPONENTS\: \{ModelType\:EstimatorSidecar" "Kepler should set desired config" || {
		return 1
	}
	return 0
}

restart_kepler() {
	kubectl delete pod -n "$NS" -l app.kubernetes.io/component=exporter || return 1
	return 0
}

restart_model_server() {
	kubectl delete pod -n "$NS" -l app.kubernetes.io/component=model-server || return 1
	return 0
}

print_usage() {
	local test_scr
	test_scr="$(basename "$0")"

	read -r -d '' help <<-EOF_HELP || true
		    🔆 Usage:
		      $test_scr
		      $test_scr     [OPTIONS]
		      $test_scr     -h|--help

		    💡 Examples:
		      # run test with estimator
		        $test_scr --estimator

		      # run test with model server
		        $test_scr --server

		      # run test with dummy weight model or power request
		        $test_scr --test

		      # enable model db
		        $test_scr --db

		      # run test estimator with model server
		        $test_scr --estimator --server

		      # run test estimator with server and model db
		        $test_scr --estimator --server --db

	EOF_HELP

	echo -e "$help"
	return 0
}

parse_args() {

	while [[ -n "${1+xxx}" ]]; do
		case $1 in
		-h | --help)
			SHOW_HELP=true
			break
			;;
		--) break ;;
		--test)
			ENABLE_TEST=true
			shift
			;;
		--estimator)
			ENABLE_ESTIMATOR=true
			shift
			;;
		--server)
			ENABLE_SERVER=true
			shift
			;;
		--db)
			ENABLE_DB=true
			shift
			;;
		*) return 1 ;;
		esac
	done
	return 0

}

patch_kepler() {
	local patch_file="$1"
	local patch="$2"
	shift 2
	[[ -n $patch_file ]] && {
		kubectl patch ds "$EXPORTER" -n "$NS" --patch-file "$patch_file" || return 1
	}
	[[ -n $patch ]] && {
		kubectl patch ds "$EXPORTER" -n "$NS" --patch "$patch" || return 1
	}
	return 0
}

patch_model_server() {
	local patch="$1"
	shift 1
	kubectl patch deploy "$MODEL_SERVER" -n "$NS" -p "$patch" || return 1
	return 0
}

patch_kepler_cfg_map() {
	kubectl patch configmap -n "$NS" "$KEPLER_CFG" --type merge -p "$(cat "$MODEL_CONFIG")"
}

run_estimator_test() {
	header "Running Estimator Test"
	if $ENABLE_TEST; then
		info "Patching power request client"
		patch_kepler "$POWER_REQUEST_CLIENT" "" || return 1
		info "Waiting for Kepler"
		wait_for_kepler || {
			return 1
		}
		info "Waiting for Kepler to get power"
		wait_for_keyword 10 10 kepler Done "cannot get power" || {
			return 1
		}
	else
		info "Checking if estimator is set and initialized"
		check_estimator_set_and_init || {
			return 1
		}
	fi

	if $ENABLE_SERVER; then
		info "Waiting for Model Server"
		wait_for_server || return 1
		info "Restarting Kepler"
		restart_kepler || return 1
		info "Waiting for Kepler"
		wait_for_kepler || return 1
		info "Waiting for estimator to load model from model server"
		wait_for_keyword 10 10 estimator "load model from model server" "estimator should be able to load model from server" || return 1

	else
		info "Waiting for estimator to load model from config"
		wait_for_keyword 10 10 estimator "load model from config" "estimator should be able to load model from config" || {
			return 1
		}
	fi

	info "Estimator Test Passed"
	line 50 heavy
	return 0
}

run_server_test() {
	header "Running Server Test"
	if $ENABLE_TEST; then
		info "Patching Kepler"
		patch_kepler "$MODEL_REQUEST_CLIENT" "" || return 1
		info "Restarting Model Server"
		restart_model_server || return 1
		info "Waiting for Kepler to get model weight"
		wait_for_keyword 10 10 kepler Done "cannot get model weight" || {
			return 1
		}
	else
		info "Waiting for Model Server"
		wait_for_server || return 1
		info "Restarting Kepler"
		restart_kepler || return 1
		info "Waiting for Kepler to get model weight"
		wait_for_keyword 10 10 kepler "getWeightFromServer.*core" "kepler should get weight from server" || {
			return 1
		}
	fi

	info "Server Test Passed"
	line 50 heavy
	return 0
}

deploy_model_db() {
	kubectl apply -f "$FILE_SERVER" || return 1
	return 0
}

print_config() {
	header "Test Configuration"
	cat <<-EOF
		  Internal Only: $ENABLE_TEST
		  Enable Estimator: $ENABLE_ESTIMATOR
		  Enable Server: $ENABLE_SERVER
		  Enable Model DB: $ENABLE_DB
		  Kepler Namespace: $NS
		  Logs Directory: $LOGS_DIR
	EOF
	line 50
}

main() {
	parse_args "$@" || exit 1

	$SHOW_HELP && {
		print_usage
		exit 0
	}

	cd "$PROJECT_ROOT"

	init_logs_dir
	print_config

	! $ENABLE_DB && {
		header "Deploying Model DB"
		deploy_model_db || {
			fail "Deploying Model DB"
			return 1
		}
		info "Waiting for Model DB"
		wait_for_db || {
			fail "Waiting for model db"
			must_gather
			return 1
		}
		info "Restarting Kepler"
		restart_kepler || {
			fail "Restarting Kepler"
			must_gather
			return 1
		}
		info "Waiting for Kepler"
		wait_for_kepler || {
			fail "Waiting for Kepler"
			must_gather
			return 1
		}

		$ENABLE_ESTIMATOR && {
			info "Patching Kepler to use Model DB"
			patch_kepler_cfg_map
			patch_kepler "" "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"estimator\",\"env\":[{\"name\":\"MODEL_TOPURL\",\"value\":\"http://model-db.kepler.svc.cluster.local:8110\"}]}]}}}}" || {
				fail "Patching kepler"
				must_gather
				return 1
			}
			info "Restarting Kepler"
			restart_kepler || {
				fail "Restarting Kepler"
				must_gather
				return 1
			}
		}
		$ENABLE_SERVER && {
			info "Patching Model Server to use Model DB"
			patch_model_server "{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"server-api\",\"env\":[{\"name\":\"MODEL_TOPURL\",\"value\":\"http://model-db.kepler.svc.cluster.local:8110\"}]}]}}}}" || return 1
			info "Restarting Model Server"
			restart_model_server || {
				fail "Restarting Model Server"
				must_gather
				return 1
			}
			info "Waiting for Model Server"
			wait_for_server || {
				fail "Waiting for model server"
				must_gather
				return 1
			}
		}
		info "Waiting for Kepler"
		wait_for_kepler || {
			fail "Waiting for kepler"
			must_gather
			return 1
		}

	}

	$ENABLE_ESTIMATOR && {
		run_estimator_test || {
			fail "Running Estimator Tests"
			must_gather
			return 1
		}
	}

	# Ensure that test are only run if only server is enabled
	[[ $ENABLE_SERVER == true && $ENABLE_ESTIMATOR != true ]] && {
		run_server_test || {
			fail "Running Server Tests"
			must_gather
			return 1
		}
	}

	return 0
}

main "$@"
