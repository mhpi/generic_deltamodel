import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt


def geoplot_single_metric(
    gdf: gpd.GeoDataFrame,
    metric_name: str,
    title: str = None,
    map_color: bool = False,
    draw_rivers: bool = False,
    dynamic_colorbar: bool = False,
    dpi: int = 100,
    marker_size: int = 50,
):
    """Geographically map a single model performance metric using Basemap.

    Parameters
    ----------
    gdf
        The GeoDataFrame containing the spatial data.
    metric_name
        The name of the metric column to plot.
    title
        The title of the plot.
    in_color
        Whether to use color for the map.
    draw_rivers
        Whether to draw rivers on the map.
    dynamic_colorbar
        Whether to use a dynamic colorbar. Default sets range from 0 to 1. This
        is not recommended for metrics that are not normalized.
    dpi
        The resolution of the plot.
    marker_size
        The size of the markers.
    """
    # Ensure the required columns are present in the GeoDataFrame
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        raise ValueError("The GeoDataFrame must include 'lat' and 'lon' columns.")

    if metric_name not in gdf.columns:
        raise ValueError(
            f"The GeoDataFrame does not contain the column '{metric_name}'.",
        )

    # Extract latitude and longitude bounds for the map
    min_lat, max_lat = (gdf['lat'].min() - 5, gdf['lat'].max())
    min_lon, max_lon = (gdf['lon'].min() - 5, gdf['lon'].max() + 5)

    # Create the figure with Cartopy
    fig, ax = plt.subplots(
        figsize=(12, 8),
        dpi=dpi,
        subplot_kw={'projection': ccrs.Mercator()},
    )
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1, linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4)

    if map_color:
        ax.add_feature(cfeature.LAND)  # , facecolor='white')
        ax.add_feature(cfeature.OCEAN)  # , facecolor='white')
    else:
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.add_feature(cfeature.OCEAN, facecolor='white')

    ax.add_feature(cfeature.COASTLINE)

    if draw_rivers:
        ax.add_feature(
            cfeature.RIVERS.with_scale('50m'),
            linewidth=0.3,
            edgecolor='blue',
        )

    # Plot the metric data
    scatter = ax.scatter(
        gdf['lon'],
        gdf['lat'],
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
    cbar.set_label(f"{metric_name.upper()}", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # Add labels and title
    plt.title(title or f"Spatial Map of {metric_name.upper()}", fontsize=14)
    plt.tight_layout()
    plt.show()
