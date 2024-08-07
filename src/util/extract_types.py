import  pandas as pd

from prom_types import  TIMESTAMP_COL, pkg_id_column
from train_types import  PowerSourceMap

container_id_colname = "id"
all_container_key = "all containers"
accelerator_type_colname = "type" 

node_level_index = [TIMESTAMP_COL]
pkg_level_index = [TIMESTAMP_COL, pkg_id_column]
container_level_index = [TIMESTAMP_COL, container_id_colname]

def component_to_col(component, unit_col=None, unit_val=None):
    power_colname = "{}_power".format(component)
    if unit_col is None:
        return power_colname
    return "{}_{}_{}".format(unit_col, unit_val, power_colname)

def col_to_component(component_col):
    splits = component_col.split('_')
    component = splits[-2:][0]
    if component == 'dynamic' or component == 'background':
        return splits[-3:][0]
    return component

def col_to_unit_val(component_col):
    return component_col.split('_')[-3:][0]

def ratio_to_col(unit_val):
    return "packge_ratio_{}".format(unit_val)

def get_unit_vals(power_columns):
    return pd.unique([col_to_unit_val(col) for col in power_columns if "package" in col])

def get_num_of_unit(energy_source, label_cols):
    energy_components = PowerSourceMap(energy_source)
    num_of_unit = len(label_cols)/len(energy_components)
    return num_of_unit

def get_expected_power_columns(energy_components, num_of_unit=1):
    # TODO: if ratio applied, 
    # return [component_to_col(component, "package", unit_val) for component in energy_components for unit_val in range(0,num_of_unit)]
    return [component_to_col(component) for component in energy_components]