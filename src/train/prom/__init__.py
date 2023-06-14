import os
import sys
cur_path = os.path.join(os.path.dirname(__file__), '.')
sys.path.append(cur_path)

from prom_query import PrometheusClient, prom_responses_to_results