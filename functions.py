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
# Necessary to supress annying matplotlib warnings
import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
from tqdm import tqdm_notebook as tqdm
import pickle
import xarray as xr
import holoviews as hv

# Input and output variable for ML parameterization
input_vars = ['air_temperature', 'specific_humidity', 'eastward_wind', 'northward_wind', 
              'air_pressure']
output_vars = [
    'air_temperature_tendency_from_convection', 
    'specific_humidity_tendency_from_convection', 
    'eastward_wind_tendency_from_convection', 
    'northward_wind_tendency_from_convection',
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
    def __init__(self, dt_seconds=1800, nx=64, ny=32, nz=10, state=None,
                 input_fields_to_store=input_vars,
                 output_fields_to_store=output_vars,
                 input_save_fn='./inputs_ref.nc',
                 output_save_fn='./outputs_ref.nc',
                 save_interval=6,
                 convection=None
                ):
        climt.set_constants_from_dict({
            'stellar_irradiance': {'value': 200, 'units': 'W m^-2'}})

        self.model_time_step = timedelta(seconds=dt_seconds)
        self.step_counter = 0
        self.save_interval = save_interval

        # Create components
        if convection is None: convection = MyEmanuelConvection()
        simple_physics = TimeDifferencingWrapper(climt.SimplePhysics())

        radiation = climt.GrayLongwaveRadiation()

        self.dycore = climt.GFSDynamicalCore(
            [simple_physics, radiation,
             convection], number_of_damped_levels=2
        )
        grid = climt.get_grid(nx=nx, ny=ny, nz=nz)
    
        if state is None:
            self.create_initial_state(grid)
        else:
            self.state = state
        
        self.state_history = [self.state]
        self.diag_history = []
        
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
        # Create model state
        self.state = climt.get_default_state([self.dycore], grid_state=grid)

        # Set initial/boundary conditions
        latitudes = self.state['latitude'].values
        longitudes = self.state['longitude'].values
        surface_shape = latitudes.shape

        temperature_equator = 300
        temperature_pole = 240

        temperature_profile = temperature_equator - (
            (temperature_equator - temperature_pole)*(
                np.sin(np.radians(latitudes))**2))

        self.state['surface_temperature'] = DataArray(
            temperature_profile*np.ones(surface_shape),
            dims=['lat', 'lon'], attrs={'units': 'degK'})
        self.state['eastward_wind'].values[:] = np.random.randn(
            *self.state['eastward_wind'].shape)
        
    def step(self):
        self.diag, self.state = self.dycore(self.state, self.model_time_step)
        self.state.update(self.diag)
        self.state['time'] += self.model_time_step
        if self.step_counter % self.save_interval == 0:
            self.input_netcdf_monitor.store(self.state)
        if (self.step_counter - 1) % self.save_interval == 0:
            self.output_netcdf_monitor.store(self.state)
        self.step_counter += 1
    
    def iterate(self, steps, noprog=False):
        for i in tqdm(range(steps), disable=noprog):
            self.step()
            
class MyEmanuelConvection(climt.EmanuelConvection):
    def __init__(self, **kwargs):
        self.diagnostic_properties = {
            'convective_state': {
                'dims': ['*'],
                'units': 'dimensionless',
                'dtype': np.int32,
            },
            'convective_precipitation_rate': {
                'dims': ['*'],
                'units': 'mm day^-1',
            },
            'convective_downdraft_velocity_scale': {
                'dims': ['*'],
                'units': 'm s^-1',
            },
            'convective_downdraft_temperature_scale': {
                'dims': ['*'],
                'units': 'degK',
            },
            'convective_downdraft_specific_humidity_scale': {
                'dims': ['*'],
                'units': 'kg/kg',
            },
            'cloud_base_mass_flux': {
                'dims': ['*'],
                'units': 'kg m^-2 s^-1',
            },
            'atmosphere_convective_available_potential_energy': {
                'dims': ['*'],
                'units': 'J kg^-1',
            },
            'air_temperature_tendency_from_convection': {
                'dims': ['*', 'mid_levels'],
                'units': 'degK s^-1',
            },
            'specific_humidity_tendency_from_convection': {
                'dims': ['*', 'mid_levels'],
                'units': 'kg/kg s^-1',
            },
            'eastward_wind_tendency_from_convection': {
                'dims': ['*', 'mid_levels'],
                'units': 'm s^-2',
            },
            'northward_wind_tendency_from_convection': {
                'dims': ['*', 'mid_levels'],
                'units': 'm s^-2',
            }
        }
        super().__init__(**kwargs)
    @ensure_contiguous_state
    def array_call(self, raw_state, timestep):
        """
        Get convective heating and moistening.
        Args:
            raw_state (dict):
                The state dictionary of numpy arrays satisfying this
                component's input properties.
        Returns:
            tendencies (dict), diagnostics (dict):
                * The heating and moistening tendencies
                * Any diagnostics associated.
        """
        self._set_fortran_constants()

        num_cols, num_levs = raw_state['air_temperature'].shape

        max_conv_level = num_levs - 3

        tendencies = initialize_numpy_arrays_with_properties(
            self.tendency_properties, raw_state, self.input_properties
        )
        diagnostics = initialize_numpy_arrays_with_properties(
            self.diagnostic_properties, raw_state, self.input_properties
        )

        q_sat = bolton_q_sat(
            raw_state['air_temperature'],
            raw_state['air_pressure'] * 100,
            self._Cpd, self._Cpv
        )

        _emanuel_convection.convect(
            num_levs,
            num_cols,
            max_conv_level,
            self._ntracers,
            timestep.total_seconds(),
            raw_state['air_temperature'],
            raw_state['specific_humidity'],
            q_sat,
            raw_state['eastward_wind'],
            raw_state['northward_wind'],
            raw_state['air_pressure'],
            raw_state['air_pressure_on_interface_levels'],
            diagnostics['convective_state'],
            diagnostics['convective_precipitation_rate'],
            diagnostics['convective_downdraft_velocity_scale'],
            diagnostics['convective_downdraft_temperature_scale'],
            diagnostics['convective_downdraft_specific_humidity_scale'],
            raw_state['cloud_base_mass_flux'],
            diagnostics['atmosphere_convective_available_potential_energy'],
            tendencies['air_temperature'],
            tendencies['specific_humidity'],
            tendencies['eastward_wind'],
            tendencies['northward_wind'])

        diagnostics['air_temperature_tendency_from_convection'][:] = (
            tendencies['air_temperature'])
        diagnostics['specific_humidity_tendency_from_convection'][:] = (
            tendencies['specific_humidity'])
        diagnostics['eastward_wind_tendency_from_convection'][:] = (
            tendencies['eastward_wind'])
        diagnostics['northward_wind_tendency_from_convection'][:] = (
            tendencies['northward_wind'])
        diagnostics['cloud_base_mass_flux'][:] = raw_state['cloud_base_mass_flux']
        return tendencies, diagnostics