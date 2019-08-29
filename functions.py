import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import climt
from climt._core import ensure_contiguous_state, bolton_q_sat
from climt._components.emanuel import _emanuel_convection
from sympl import (
    PlotFunctionMonitor,
    TimeDifferencingWrapper,
    DataArray,
    DiagnosticComponent,
    initialize_numpy_arrays_with_properties,
    NetCDFMonitor,
    TendencyComponent
)
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import pickle
import xarray as xr
import holoviews as hv

# Input and output variable for ML parameterization
input_vars = ['air_temperature', 'specific_humidity', 'eastward_wind', 'northward_wind', 
              'air_pressure']
output_vars = [
    'air_temperature_tendency_from_EmanuelConvection', 
    'specific_humidity_tendency_from_EmanuelConvection', 
    'eastward_wind_tendency_from_EmanuelConvection',
    'northward_wind_tendency_from_EmanuelConvection',
    'convective_precipitation_rate'
]


def make_animation(sliced_data, cmap_range=None, **kwargs):
    hv_ds = hv.Dataset(
        (np.arange(64), np.arange(32), 
         np.arange(len(sliced_data.time)), sliced_data),
        ['lon', 'lat', 'time'], sliced_data.name
    )
    hv_img = hv_ds.to(hv.Image, ['lon', 'lat']).options(**kwargs)
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(**{var_name: hv.Dimension(var_name, range=cmap_range)})
    return hv_img

def normalize(arr, mean, std):
    return (arr - mean) / std
def unnormalize(arr, mean, std):
    return arr * std + mean

input_vars = ['air_temperature', 'specific_humidity', 'eastward_wind', 'northward_wind', 
              'air_pressure']
output_vars = [
    'air_temperature_tendency_from_convection', 
    'specific_humidity_tendency_from_convection', 
    'eastward_wind_tendency_from_convection', 
    'northward_wind_tendency_from_convection',
    'convective_precipitation_rate'
]

class MyModel():
    """Climate model class."""
    def __init__(self, dt_seconds=1800, nx=64, ny=32, nz=10, state=None,
                 input_fields_to_store=input_vars,
                 output_fields_to_store=output_vars,
                 input_save_fn='./inputs_ref.nc',
                 output_save_fn='./outputs_ref.nc',
                 save_interval=6,
                 convection=None
                ):
        """
        Initialize model. Uses SSTs from Andersen and Kuang 2012.
        Creates initial state unless state is given.
        """
        climt.set_constants_from_dict({
            'stellar_irradiance': {'value': 200, 'units': 'W m^-2'}})

        self.model_time_step = timedelta(seconds=dt_seconds)
        self.step_counter = 0
        self.save_interval = save_interval

        # Create components
        if convection is None: 
            convection = climt.EmanuelConvection(tendencies_in_diagnostics=True)
        simple_physics = TimeDifferencingWrapper(
            climt.SimplePhysics(tendencies_in_diagnostics=True))

        radiation = climt.GrayLongwaveRadiation(
            tendencies_in_diagnostics=True)

        self.dycore = climt.GFSDynamicalCore(
            [simple_physics, radiation,
             convection], number_of_damped_levels=2
        )
        grid = climt.get_grid(nx=nx, ny=ny, nz=nz)
    
        if state is None:
            self.create_initial_state(grid)
        else:
            self.state = state
        
        self.input_netcdf_monitor = NetCDFMonitor(
            input_save_fn,
            write_on_store=True,
            store_names=input_fields_to_store
        )
        self.output_netcdf_monitor = NetCDFMonitor(
            output_save_fn,
            write_on_store=True,
            store_names=output_fields_to_store
        )
        
    def create_initial_state(self, grid):
        """Create initial state."""
        # Create model state
        self.state = climt.get_default_state([self.dycore], grid_state=grid)

        # Set initial/boundary conditions
        latitudes = self.state['latitude'].values
        sst_k = and_kua_sst(latitudes)
        
        self.state['surface_temperature'] = DataArray(sst_k,
            dims=['lat', 'lon'], attrs={'units': 'degK'})
        self.state['eastward_wind'].values[:] = np.random.randn(
            *self.state['eastward_wind'].shape)
        
    def step(self):
        """Take one time step forward."""
        self.diag, self.state = self.dycore(self.state, self.model_time_step)
        self.state.update(self.diag)
        self.state['time'] += self.model_time_step
        if self.step_counter % self.save_interval == 0:
            self.input_netcdf_monitor.store(self.state)
        if (self.step_counter - 1) % self.save_interval == 0:
            self.output_netcdf_monitor.store(self.state)
        self.step_counter += 1
    
    def iterate(self, steps, noprog=False):
        """Iterate over several time steps."""
        for i in tqdm(range(steps), disable=noprog):
            self.step()
            
def and_kua_sst(latitudes):
    zeta = np.ones_like(latitudes)
    lat_range = (latitudes > 5) & (latitudes <= 60)
    zeta[lat_range] = (np.sin(np.pi * (latitudes - 5) / 110)**2)[lat_range]
    lat_range = (latitudes >= -60) & (latitudes <= 5)
    zeta[lat_range] = (np.sin(np.pi * (latitudes - 5) / 130)**2)[lat_range]
    sst_c = 2 + 27/2*(2 - zeta - zeta**2)
    sst_k = sst_c + 273.15
    return sst_k
            