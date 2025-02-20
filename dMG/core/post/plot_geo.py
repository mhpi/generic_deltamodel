import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def geoplot_single_metric(
    gdf: gpd.GeoDataFrame,
    metric_name: str,
    title: str = None,
    draw_rivers: bool = False,
    dynamic_colorbar: bool = False,
    marker_size: int = 50,
):
    """Geographically map a single model performance metric using Basemap.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame containing the spatial data.
    metric_name : str
        The name of the metric column to plot.
    title : str, optional
        The title of the plot. Default is None.
    draw_rivers : bool, optional
        Whether to draw rivers on the map. Default is False.
    dynamic_colorbar : bool, optional
        Whether to use a dynamic colorbar. Default is False, setting range from
        0 to 1. This is not recommended for metrics that are not normalized.
    marker_size : int, optional
        The size of the markers. Default is 50.
    """
    # Ensure the required columns are present in the GeoDataFrame
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        raise ValueError("The GeoDataFrame must include 'lat' and 'lon' columns.")
    
    if metric_name not in gdf.columns:
        raise ValueError(f"The GeoDataFrame does not contain the column '{metric_name}'.")
    
    # Extract latitude and longitude bounds for the map
    min_lat, max_lat = gdf['lat'].min() - 5, gdf['lat'].max() + 5
    min_lon, max_lon = gdf['lon'].min() - 5, gdf['lon'].max() + 5
    
    # Create a Basemap instance
    plt.figure(figsize=(12, 8))
    m = Basemap(
        llcrnrlon=min_lon, llcrnrlat=min_lat,  # Lower-left corner
        urcrnrlon=max_lon, urcrnrlat=max_lat,  # Upper-right corner
        resolution='i',  # Intermediate resolution
        projection='merc',  # Mercator projection
        lat_0=(min_lat + max_lat) / 2, lon_0=(min_lon + max_lon) / 2
    )
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.drawstates(linewidth=0.2)

    if draw_rivers:
        m.drawrivers(linewidth=0.1, color='blue')

    m.drawmapboundary(fill_color='white')
    
    # Convert lat/lon to map projection coordinates
    x, y = m(gdf['lon'].values, gdf['lat'].values)
    
    # Plot the scatter points
    if dynamic_colorbar:
        scatter = m.scatter(
            x, y,
            c=gdf[metric_name],
            cmap='coolwarm',
            s=marker_size,  # Marker size
            alpha=0.95,  # Transparency
            edgecolor='k'  # Black border around markers
        )
    else:
        scatter = m.scatter(
            x, y,
            c=gdf[metric_name],
            cmap='coolwarm',
            s=marker_size,  # Marker size
            alpha=0.95,  # Transparency
            edgecolor='k',  # Black border around markers
            vmin=0,  # Minimum value for the color scale
            vmax=1   # Maximum value for the color scale
        )    
        
    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05)
    cbar.set_label(f"{metric_name.upper()}", fontsize=12)  

    # Add labels and title
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f"Spatial Map of {metric_name.upper()}", fontsize=14)
    
    # Show the plot
    plt.show()