import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

with rasterio.open("phl_ppp_2020.tif") as src:
    dst_crs = "EPSG:32651"  # UTM 51N
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({"crs": dst_crs, "transform": transform,
                   "width": width, "height": height})

    with rasterio.open("phl_pop_utm51.tif", "w", **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=dst_crs,
            resampling=Resampling.sum  # sum, not average — preserves population counts
        )