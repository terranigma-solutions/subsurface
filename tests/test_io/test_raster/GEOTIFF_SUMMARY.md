# GeoTIFF Test Data Summary

## How `read_structured_topography` handles each file

`read_structured_topography(path, crop_to_extent=None, band=None)` is the single entry point.
- **`crop_to_extent`** — `[left, bottom, right, top]` in raster CRS coords to read a sub-window.
- **`band`** — explicit band index (1-based). If `None`, auto-selects the richest band (most unique values, widest range, most valid data).

---

### Single-channel (grayscale / elevation) — defaults are sufficient

| File | Dtype | Bands | Notes | Special args? |
|---|---|---|---|---|
| `geotiff-testdata-lisbon-elevation.tif` | Float32 | 1 | EPSG:4326 (geographic) — auto-projected to UTM | No |
| `geotiff-testdata-wind-direction.tif` | Float32 | 1 | LZW compressed, EPSG:4326 | No |
| `osgeo-usgs-i30dem.tif` | Int16 (signed) | 1 | EPSG:26710 (UTM NAD27), **PixelIsPoint** — coords are at pixel corners, but reader treats as pixel-center | No |
| `osgeo-gdal-cea.tif` | UInt8 | 1 | User-defined Cylindrical Equal Area CRS | No |
| `tiff1.tif` | Float32 | 1 | EPSG:3034 (LCC Europe), NoData=nan | No |
| `ET_DTM_HC_2026_FINAL_50m.tif` | Float32 | 1 | EPSG:3034 (LCC Europe), NoData=nan | No |
| `geotiff-testdata-vanderford-cog.tif` | Float32 | 1 | EPSG:3031 (Antarctic Polar Stereographic), Deflate compression, tiled, **COG with overviews (4 IFDs)** — only the main IFD is read; overviews are ignored | No |

### Multi-channel (RGB/RGBA) — auto-selects the richest band

| File | Dtype | Bands | Notes | Special args? |
|---|---|---|---|---|
| `rasterio-rgb-byte.tif` | UInt8 | 3 (RGB) | EPSG:32618 (UTM WGS84), uncompressed, stripped | No — auto-selects the band with most info (typically band 2 or 3) |
| `geotiff-testdata-gadas.tif` | UInt8 | 4 (RGBA + unassociated alpha) | Web Mercator, uncompressed | No — auto-selects band, alpha band is skipped if it's all-constant |
| `tiff2.tiff` | UInt8 | 4 (RGBA) | EPSG:3034 (LCC Europe), LZW compressed, **INTERLEAVE=PIXEL** | No |
| `tiff3.tiff` | UInt8 | 4 (RGBA) | Same as tiff2 but **12699 × 11797** (large) | No — but expect longer I/O |

### Palette — decoded by rasterio to a single-channel array

| File | Dtype | Bands | Notes | Special args? |
|---|---|---|---|---|
| `geotiff-testdata-geogtowgs84.tif` | UInt8 (palette) | 1 (indexed) | PackBits compression, has `ColorMap` tag, `GeogTOWGS84GeoKey` datum transform | No — rasterio decodes the palette, reader gets a single-channel Float32 array |

---

## Quick reference — band recipes

| Use case | Argument |
|---|---|
| Read a specific band from a multi-channel file | `band=1`, `band=2`, etc. |
| Let the reader pick the best band | Default (omit `band`) |
| Crop to region of interest | `crop_to_extent=[xmin, ymin, xmax, ymax]` |

All files above can be imported with defaults — no special arguments strictly required.