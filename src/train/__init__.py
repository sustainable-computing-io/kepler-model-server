# comonly used within train module

import os
import sys
cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)
util_path = os.path.join(os.path.dirname(__file__), '..', 'util')
sys.path.append(util_path)
extractor_path = os.path.join(os.path.dirname(__file__), 'extractor')
sys.path.append(extractor_path)
isolator_path = os.path.join(os.path.dirname(__file__), 'isolator')
sys.path.append(isolator_path)
profiler_path = os.path.join(os.path.dirname(__file__), 'profiler')
sys.path.append(profiler_path)

from extractor import DefaultExtractor
from profiler import Profiler, generate_profiles
from isolator import MinIdleIsolator, ProfileBackgroundIsolator, NoneIsolator
from train_isolator import TrainIsolator
from pipeline import NewPipeline, load_class

DefaultProfiler = Profiler(extractor=DefaultExtractor())