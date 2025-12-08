
import geopandas as gpd
from pathlib import Path
import zarr
import xarray as xr
import json
import pandas as pd
import networkx as nx
import time


hf_path = '/nfs/data/wby5078/CONUS3000/conus_nextgen.gpkg'
divides = gpd.read_file(Path(hf_path), layer="divides")
divides["divide_id"] = divides["divide_id"].str.extract(r'(\d+)$')[0].astype(int)


#----------------------------------- Texas ------------------------------------#

texas_runoff_utc = '/data/wby5078/raw_data/texas_raw/hourly_data/discharge_processed_utc0'
gauge_info_path = '/nfs/data/wby5078/Texas_floods/info_stations_TX_595.csv'
area_gap_thres = 0.2

gauges = zarr.open_group(Path(texas_runoff_utc))['GAGEID'][:]
gauge_info = pd.read_csv(Path('/nfs/data/wby5078/Texas_floods/') / 'info_stations_TX_595.csv')
gauge_info['gauge_id'] = gauge_info['STAID'].astype(str).str.zfill(8)

### get gauge hf
gauge_gdf = gpd.GeoDataFrame(
    gauge_info,
    geometry=gpd.points_from_xy(gauge_info['LNG_GAGE'], gauge_info['LAT_GAGE']),
    crs="EPSG:4326"
).to_crs(5070)
gauge_gdf = gpd.sjoin(gauge_gdf, divides, predicate="intersects", how="left")
gauge_gdf = gauge_gdf[gauge_gdf['gauge_id'].isin(gauges) & ((gauge_gdf['DRAIN_SQKM'] - gauge_gdf['tot_drainage_areasqkm']).abs() / gauge_gdf['DRAIN_SQKM'] < area_gap_thres)].reset_index(drop=True)

#----------------------------------- CAMELS ------------------------------------#

area_gap_thres = 0.2

attrs = xr.open_dataset(Path('/data/wby5078/data/diffusion-ds/input/') / 'attrs.nc')
gauge_info = pd.read_csv(Path('/nfs//data/wby5078/Camels_hourly/') / 'camels_topo.txt', sep=";")
gauge_info['gauge_id'] = gauge_info['gauge_id'].astype(str).str.zfill(8)
gauge_info = gauge_info[gauge_info['gauge_id'].isin(attrs['gauge'].data)].reset_index(drop=True)

### get gauge hf
gauge_gdf = gpd.GeoDataFrame(
    gauge_info,
    geometry=gpd.points_from_xy(gauge_info['gauge_lon'], gauge_info['gauge_lat']),
    crs="EPSG:4326"
).to_crs(5070)
gauge_gdf = gpd.sjoin(gauge_gdf, divides, predicate="intersects", how="left")
gauge_gdf = gauge_gdf[(gauge_gdf['area_gages2'] - gauge_gdf['tot_drainage_areasqkm']).abs() / gauge_gdf['area_gages2'] < area_gap_thres].reset_index(drop=True)

#----------------------------------- CONUS/CAMELSH ------------------------------------#

area_gap_thres = 0.2

df_avail = pd.read_csv(Path('/nfs/data/wby5078/CAMELSH/') / 'info.csv', dtype={'STAID': str})
pts = gpd.read_file(Path('/nfs/data/wby5078/CONUS3000/gagesII_9322_sept30_2011/') / 'gagesII_9322_sept30_2011.shp')
pts = pts[pts['STAID'].isin(df_avail.loc[df_avail['data_availability [hrs]'] > 0, 'STAID'])].reset_index(drop=True)
gauge_info = pts[['STAID', 'LAT_GAGE', 'LNG_GAGE', 'DRAIN_SQKM']].rename(columns={'STAID': 'gauge_id', 'LAT_GAGE': 'lat', 'LNG_GAGE': 'lon'})

### get gauge hf
gauge_gdf = gpd.GeoDataFrame(
    gauge_info,
    geometry=gpd.points_from_xy(gauge_info['lon'], gauge_info['lat']),
    crs="EPSG:4326"
).to_crs(5070)
gauge_gdf = gpd.sjoin(gauge_gdf, divides, predicate="intersects", how="left")
gauge_gdf = gauge_gdf[(gauge_gdf['DRAIN_SQKM'] - gauge_gdf['tot_drainage_areasqkm']).abs() / gauge_gdf['DRAIN_SQKM'] < area_gap_thres].reset_index(drop=True)

#---------------------------- Save lookup -------------------------------#

### create network
# save_path = '/nfs/data/wby5078/Camels_hourly/gage_hf.json'
save_path = '/nfs/data/wby5078/CONUS3000/gage_hf.json'

hf = gpd.read_file(Path(hf_path))
df_edges = pd.DataFrame({'ids': hf.id.str.extract(r'(\d+)$')[0].astype(int), 'toids': hf.toid.str.extract(r'(\d+)$')[0].astype(int)})
df_edges = df_edges.drop_duplicates().reset_index(drop=True)
G = nx.from_pandas_edgelist(df_edges, source='ids', target='toids', create_using=nx.DiGraph)
all_divides = set()
for gid in gauge_gdf['gauge_id']:
    divide_id = gauge_gdf[gauge_gdf['gauge_id'] == gid]['divide_id'].values[0]
    if pd.isna(divide_id):
        continue
    ancestors = nx.ancestors(G, divide_id)
    ancestors.add(int(divide_id))
    all_divides.update(ancestors)
G = G.subgraph(all_divides).copy()
gauge_df_dict = {
    'nodes': list(G.nodes),
    'edges': list(G.edges),
    'gage_hf': dict(zip(gauge_gdf['gauge_id'], gauge_gdf['divide_id']))
}
json.dump(gauge_df_dict, open(Path(save_path), "w"))


