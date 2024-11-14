import ecmwf.data as ecdata
from ecmwf.opendata import Client
import metview as mv


parameters = ['tp']
filename = 'data/precipitation/medium-tp-mean.grib2'


def get_fieldset():
    data = mv.read(filename)
    return data


def interpolate_precipitation(fieldset, df):
    df = df.copy()
    interpolated_data = mv.interpolate(fieldset, df['lat'], df['lon'])
    df['precipitation'] = interpolated_data
    return df
