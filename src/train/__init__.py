# comonly used within train module

import os
import sys

cur_path = os.path.join(os.path.dirname(__file__), ".")
sys.path.append(cur_path)
util_path = os.path.join(os.path.dirname(__file__), "..", "util")
sys.path.append(util_path)
extractor_path = os.path.join(os.path.dirname(__file__), "extractor")
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), "isolator")
sys.path.append(isolator_path)

from extractor import DefaultExtractor
from isolator import MinIdleIsolator, NoneIsolator, ProfileBackgroundIsolator
from pipeline import NewPipeline, load_class
from profiler.node_type_index import NodeTypeIndexCollection, NodeTypeSpec
from profiler.profiler import Profiler, generate_profiles
from smooth_extractor import SmoothExtractor
from train_isolator import TrainIsolator

DefaultProfiler = Profiler(extractor=DefaultExtractor())
