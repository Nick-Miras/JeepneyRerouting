import ecmwf.data as ecdata
from magpye import GeoMap
from ecmwf.opendata import Client
import xarray as xr
import osmnx as ox
import joblib

client = Client("ecmwf")
model = joblib.load('')

# total precipitation probability
parameters = ['tpg1', 'tpg5', 'tpg10', 'tpg20', 'tpg25', 'tpg50', 'tpg100']
precipitation_thresholds = [1, 5, 10, 20, 25, 50, 100]

filename = 'precipitation/tp-probabilities.grib'

client.retrieve(
    step=[i for i in range(24, 48, 3)],
    stream="enfo",
    type=['cf', 'pf'],
    levtype="sfc",
    param=parameters,
    target=filename
)

def simulate_flooding(graph):
    


if __name__ == '__main__':
    graph_path = ''
    route_graph = ox.io.load_graphml(graph_path)
    