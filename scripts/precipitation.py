import ecmwf.data as ecdata
from magpye import GeoMap
from ecmwf.opendata import Client
import metview as mv

client = Client("ecmwf", beta=True)

# total precipitation probability
parameters = ['tpg1', 'tpg5' ,'tpg10' ,'tpg20' ,'tpg25' ,'tpg50' ,'tpg100']
filename = 'precipitation/medium-tp-mean.grib'

client.retrieve(
    step="24-48",  # get the forecast for tomorrow
    stream="enfo",
    type="ep",
    levtype="sfc",
    param=parameters,
    target=filename
)

data = mv.read(filename)


def interpolate_precipitation(fieldset, df):
    df = df.copy()
    interpolated_data = mv.interpolate(fieldset, data['lat'], data['lon'])
    df['precipitation'] = interpolated_data
    return df