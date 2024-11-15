import ecmwf.data as ecdata
from magpye import GeoMap
from ecmwf.opendata import Client
import xarray as xr
import osmnx as ox
import joblib
import typing
import geopandas as gpd
import networkx as nx
import numpy as np

client = Client("ecmwf")
model = joblib.load('../models/rf-model.joblib')
umap_reducer = joblib.load('../models/euclidian-umap-reducer.joblib')
dem = xr.open_dataset('../graphs/dem.nc')
THRESHOLD = 0.01

flood_hazard_map_5yr = gpd.read_file('../noah_data/5yr/MetroManila5yr.zip', crs='EPSG:32633')
flood_hazard_map_25yr = gpd.read_file('../noah_data/25yr/MetroManila25yr.zip', crs='EPSG:32633')
flood_hazard_map_100yr = gpd.read_file('../noah_data/100yr/MetroManila100yr.zip', crs='EPSG:32633')

risk_levels = [0.25, 0.05, 0.01]
hazard_maps = [flood_hazard_map_5yr, flood_hazard_map_25yr, flood_hazard_map_100yr]

parameters = ['tp']
steps = [i for i in range(24, 48, 3)]

FILENAME = 'data/precipitation/tp-probabilities.grib'

def get_elevation(dem, lat, lon):
    # elevation = dem.interp(lat=lat, lon=lon, method='linear')
    # return elevation['z'].values
    return 0
    
def retrieve_data(client):
    client.retrieve(
        time=0,
        step=[i for i in range(24, 48, 3)],
        stream="enfo",
        type=['pf'],
        levtype="sfc",
        param=parameters,
        target=FILENAME
    )

def get_hazard_level(lat, lon):
    for hazard_map, risk_level in zip(hazard_maps, risk_levels):
        if hazard_map.to_crs(epsg=4326).geometry.contains(Point(lon, lat))[0] is True:
            return risk_level

def interp_data(data, target_lon, target_lat):
    return data.interp(longitude=target_lon, latitude=target_lat, method='linear')

def transform_to_x(graph, data) -> typing.Generator:
    for node_id, data in graph.nodes(data=True):
        lon, lat = (data.get('lon'), data.get('lat'))
        precipitation = interp_data(data, lon, lat)
        yield np.array(lat, lon, precipitation['tp'].values, get_hazard_level(lat, lon), get_elevation(dem, lat, lon)), node_id

def get_flooded_nodes(graph, data) -> typing.Generator:
    array = np.array(list(transform_to_x(graph, data)))
    x = array[:, 0]
    node_ids = array[:, 1]
    predictions = model.predict(x)
    predictions = (predictions >= THRESHOLD).astype(int)
    for i, prediction in enumerate(predictions):
        if prediction == 1:
            yield node_ids[0]

def simulate_flooding(graph: nx.Graph, data):
    return graph.remove_nodes_from(list(get_flooded_nodes(graph, data)))



if __name__ == '__main__':
    retrieve_data(client)
    data = xr.load_dataset(FILENAME, engine='cfgrib')
    graph_path = '../graphs/graphml/jeepneyroutes/T307-graph-original.graphml'
    route_graph = ox.io.load_graphml(graph_path)
    for step, group in data.groupby('step'):
        filename_prefix = 'map0-nodes'
        graph = route_graph.copy()
        df = ox.graph_to_gdfs(simulate_flooding(graph, group))
        df.to_file(filename_prefix + '-' + step / 3, driver='GeoJSON')
        