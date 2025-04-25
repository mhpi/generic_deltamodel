import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt


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
    
    # Create the figure with Cartopy
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Mercator()})
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.2)
    ax.add_feature(cfeature.BACKGROUND())
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    if draw_rivers:
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.3, edgecolor='blue')

    # Plot the metric data
    scatter = ax.scatter(
        gdf['lon'], gdf['lat'],
        c=gdf[metric_name],
        s=marker_size,
        cmap='coolwarm',
        alpha=0.95,
        edgecolor='k',
        transform=ccrs.PlateCarree(),
        vmin=(None if dynamic_colorbar else 0),
        vmax=(None if dynamic_colorbar else 1),
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.05, ax=ax)
    cbar.set_label(f"{metric_name.upper()}", fontsize=12)

    # Add labels and title
    plt.title(title or f"Spatial Map of {metric_name.upper()}", fontsize=14)
    plt.tight_layout()
    plt.show()
