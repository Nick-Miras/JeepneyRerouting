import xarray as xr


def load_dataset(filename):
    return xr.load_dataset(filename, engine='cfgrib')

def interp_data(data, target_lon, target_lat):
    return data.interp(longitude=target_lon, latitude=target_lat, method='linear')

def get_precipitation(data, target_lon, target_lat):
    interped_data = interp_data(data, target_lon, target_lat)
    return interped_data
