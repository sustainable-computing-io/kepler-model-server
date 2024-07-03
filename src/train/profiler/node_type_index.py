# NodeTypeIndexCollection manages node_type index under pipeline path
# p
# Usage:
#   index_collection = NodeTypeIndexCollection(pipeline_path)
#   node_type = index_collection.index_train_machine(processor, cores, chips, cache_kb, memory_gb, cpu_freq_mhz, power_mgt, minmax_watt)
#   index_collection.save()

import os
import sys

util_path = os.path.join(os.path.dirname(__file__), "..", "..", "util")
sys.path.append(util_path)

import enum
import re

from loader import load_node_type_index
from saver import save_machine_spec, save_node_type_index


def rename(name):
    name = name.replace("(R)", "")
    name = name.replace("(r)", "")
    name = name.replace("CPU", "")
    name = name.replace("Processor", "")
    name = name.replace("processor", "")
    name = re.sub(r"\d+-Bit Multi-Core", "", name)
    name = name.split("(")[0].strip()
    name = name.split("[")[0].strip()
    name = name.replace("@", "")
    name = name.replace("®", "")
    name = name.replace(",", "")
    name = re.sub(r"\d+(\.\d+)?\s?[G|M|g|m][H|h]z", "", name).strip()
    return name


def format_processor(processor):
    return "_".join(re.sub(r"\(.*\)", "", rename(processor)).split()).replace("-", "_").lower().replace("_v", "v")


def format_vendor(vendor):
    return vendor.split()[0].replace("-", "_").replace(",", "").replace("'", "").lower()


GB = 1024 * 1024 * 1024
import subprocess

import cpuinfo
import psutil
import pyudev


def generate_spec(data_path, machine_id):
    processor = ""
    vendor = ""
    cpu_info = cpuinfo.get_cpu_info()
    if "brand_raw" in cpu_info:
        processor = format_processor(cpu_info["brand_raw"])
    context = pyudev.Context()
    for device in context.list_devices(subsystem="dmi"):
        if device.get("ID_VENDOR") is not None:
            vendor = format_vendor(device.get("ID_VENDOR"))
            break
    cores = psutil.cpu_count(logical=True)
    chips = max(1, int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True)))
    threads_per_core = max(1, cores // psutil.cpu_count(logical=False))
    memory = psutil.virtual_memory().total
    memory_gb = int(memory / GB)
    freq = psutil.cpu_freq(percpu=False)
    cpu_freq_mhz = round(max(freq.max, freq.current) / 100) * 100  # round to one decimal of GHz
    spec_values = {
        "vendor": vendor,
        "processor": processor,
        "cores": cores,
        "chips": chips,
        "memory_gb": memory_gb,
        "cpu_freq_mhz": cpu_freq_mhz,
        "threads_per_core": threads_per_core,
    }
    spec = NodeTypeSpec(**spec_values)
    print(f"Save machine spec ({data_path}): ")
    print(str(spec))
    save_machine_spec(data_path, machine_id, spec)


class NodeAttribute(str, enum.Enum):
    PROCESSOR = "processor"
    CORES = "cores"
    CHIPS = "chips"
    MEMORY = "memory"
    FREQ = "frequency"


def load_node_type_spec(node_type_index_json):
    node_type_spec_index = dict()
    if node_type_index_json is not None:
        for index, spec_obj in node_type_index_json.items():
            spec = NodeTypeSpec()
            spec.load(spec_obj)
            node_type_spec_index[int(index)] = spec
    return node_type_spec_index


no_data = None


# NodeTypeSpec defines spec of each node_type index
class NodeTypeSpec:
    def __init__(self, **kwargs):
        self.attrs = dict()
        self.attrs[NodeAttribute.PROCESSOR] = kwargs.get("processor", no_data)
        self.attrs[NodeAttribute.CORES] = kwargs.get("cores", no_data)
        self.attrs[NodeAttribute.CHIPS] = kwargs.get("chips", no_data)
        self.attrs[NodeAttribute.MEMORY] = kwargs.get("memory_gb", no_data)
        self.attrs[NodeAttribute.FREQ] = kwargs.get("cpu_freq_mhz", no_data)
        self.members = []

    def load(self, json_obj):
        for attr, attr_values in json_obj["attrs"].items():
            self.attrs[attr] = attr_values
        self.members = json_obj["members"]

    def add_member(self, machine_id):
        if machine_id in self.members:
            print("member already exists: ", machine_id)
            return True
        self.members += [machine_id]
        return True

    def get_size(self):
        return len(self.members)

    def get_cores(self):
        return self.attrs[NodeAttribute.CORES]

    # check the comparing node-type spec is covered by this node-type spec
    def cover(self, compare_spec):
        if not isinstance(compare_spec, NodeTypeSpec):
            return False
        for attr in NodeAttribute:
            if compare_spec.attrs[attr] is not None:
                try:
                    # Attempt to convert values to floats
                    if float(self.attrs[attr]) != float(compare_spec.attrs[attr]):
                        return False
                except ValueError:
                    # If conversion to float fails, compare as strings
                    if self.attrs[attr] != compare_spec.attrs[attr]:
                        return False
        return True

    def __str__(self):
        out_str = ""
        for attr in NodeAttribute:
            out_str += f"{attr} ({self.attrs[attr]!s})\n"
        return out_str

    def get_json(self):
        json_obj = dict()
        json_obj["attrs"] = dict()
        for attr in NodeAttribute:
            json_obj["attrs"][f"{attr}"] = self.attrs[attr]
        json_obj["members"] = self.members
        return json_obj

    def complete_info(self):
        for attr in NodeAttribute:
            if self.attrs[attr] is None:
                return False
        return True

    def copy(self):
        spec = NodeTypeSpec()
        spec.attrs = self.attrs.copy()
        spec.members = self.members.copy()
        return spec


class NodeTypeIndexCollection:
    def __init__(self, pipeline_path):
        self.pipeline_path = pipeline_path
        node_type_index_json = load_node_type_index(self.pipeline_path)
        self.node_type_index = load_node_type_spec(node_type_index_json)

    def index_train_machine(self, machine_id, new_spec):
        if not new_spec.complete_info():
            print("Machine info not completed: ", str(new_spec))
            return -1
        covered_index = self.get_node_type(new_spec)
        if covered_index == -1:
            covered_index = 0
            if len(self.node_type_index.keys()) > 0:
                covered_index = max(self.node_type_index.keys()) + 1
            self.node_type_index[covered_index] = new_spec
        self.node_type_index[covered_index].add_member(machine_id)
        return covered_index

    def get_node_type(self, compare_spec):
        if len(self.node_type_index) == 0:
            return -1
        for index, node_type_spec in self.node_type_index.items():
            if node_type_spec.cover(compare_spec):
                return index
        return -1

    def get_json(self):
        json_obj = dict()
        for index, node_type_spec in self.node_type_index.items():
            json_obj[index] = node_type_spec.get_json()
        return json_obj

    def save(self):
        obj = self.get_json()
        save_node_type_index(self.pipeline_path, obj)

    def copy(self):
        node_collection = NodeTypeIndexCollection(self.pipeline_path)
        removed_items = [
            node_type
            for node_type in node_collection.node_type_index.keys()
            if node_type not in self.node_type_index.keys()
        ]
        for node_type in removed_items:
            del node_collection.node_type_index[node_type]
        return node_collection
