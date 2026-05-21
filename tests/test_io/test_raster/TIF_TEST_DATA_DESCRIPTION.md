# Create GeoTIFF sample data

Owner: Aaron, Miguel , Simon Virgo
Status: Done
Assign: Aaron
Doing date: May 18, 2026 → May 22, 2026
Sprints: Sprint T79, Sprint T80
Pin: No
ID: GEN-15151
Parent: Import GeoTiff (1) (https://www.notion.so/Import-GeoTiff-1-32da955ff055807fa32eed1aaa7af9fa?pvs=21)
Ideon status: Not ported
QA Status: Info needed
Scrum Inbox: Miguel
Scrum Team: Tech

Exported from the Einstein Telescope project:

- Tiff1


  ```
  Driver: GTiff/GeoTIFF
  Files: tiff1.tif
  Size is 1408, 1405
  Coordinate System is:
  PROJCRS["ETRS89-extended / LCC Europe",
      BASEGEOGCRS["ETRS89",
          ENSEMBLE["European Terrestrial Reference System 1989 ensemble",
              MEMBER["European Terrestrial Reference Frame 1989"],
              MEMBER["European Terrestrial Reference Frame 1990"],
              MEMBER["European Terrestrial Reference Frame 1991"],
              MEMBER["European Terrestrial Reference Frame 1992"],
              MEMBER["European Terrestrial Reference Frame 1993"],
              MEMBER["European Terrestrial Reference Frame 1994"],
              MEMBER["European Terrestrial Reference Frame 1996"],
              MEMBER["European Terrestrial Reference Frame 1997"],
              MEMBER["European Terrestrial Reference Frame 2000"],
              MEMBER["European Terrestrial Reference Frame 2005"],
              MEMBER["European Terrestrial Reference Frame 2014"],
              MEMBER["European Terrestrial Reference Frame 2020"],
              ELLIPSOID["GRS 1980",6378137,298.257222101,
                  LENGTHUNIT["metre",1]],
              ENSEMBLEACCURACY[0.1]],
          PRIMEM["Greenwich",0,
              ANGLEUNIT["degree",0.0174532925199433]],
          ID["EPSG",4258]],
      CONVERSION["Europe Conformal 2001",
          METHOD["Lambert Conic Conformal (2SP)",
              ID["EPSG",9802]],
          PARAMETER["Latitude of false origin",52,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8821]],
          PARAMETER["Longitude of false origin",10,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8822]],
          PARAMETER["Latitude of 1st standard parallel",35,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8823]],
          PARAMETER["Latitude of 2nd standard parallel",65,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8824]],
          PARAMETER["Easting at false origin",4000000,
              LENGTHUNIT["metre",1],
              ID["EPSG",8826]],
          PARAMETER["Northing at false origin",2800000,
              LENGTHUNIT["metre",1],
              ID["EPSG",8827]]],
      CS[Cartesian,2],
          AXIS["northing (N)",north,
              ORDER[1],
              LENGTHUNIT["metre",1]],
          AXIS["easting (E)",east,
              ORDER[2],
              LENGTHUNIT["metre",1]],
      USAGE[
          SCOPE["Conformal mapping at scales of 1:500,000 and smaller."],
          AREA["Europe - European Union (EU) countries and candidates. Europe - onshore and offshore: Albania; Andorra; Austria; Belgium; Bosnia and Herzegovina; Bulgaria; Croatia; Cyprus; Czechia; Denmark; Estonia; Faroe Islands; Finland; France; Germany; Gibraltar; Greece; Hungary; Iceland; Ireland; Italy; Kosovo; Latvia; Liechtenstein; Lithuania; Luxembourg; Malta; Monaco; Montenegro; Netherlands; North Macedonia; Norway including Svalbard and Jan Mayen; Poland; Portugal including Madeira and Azores; Romania; San Marino; Serbia; Slovakia; Slovenia; Spain including Canary Islands; Sweden; Switzerland; Türkiye (Turkey); United Kingdom (UK) including Channel Islands and Isle of Man; Vatican City State."],
          BBOX[24.6,-35.58,84.73,44.83]],
      ID["EPSG",3034]]
  Data axis to CRS axis mapping: 2,1
  Origin = (3684217.673899999819696,2702950.319999999832362)
  Pixel Size = (50.010753693181783,-49.993288469750738)
  Metadata:
    AREA_OR_POINT=Area
  Image Structure Metadata:
    INTERLEAVE=BAND
  Corner Coordinates:
  Upper Left  ( 3684217.674, 2702950.320) (  5d20'16.45"E, 51d 0'14.31"N)
  Lower Left  ( 3684217.674, 2632709.750) (  5d24' 8.43"E, 50d21' 4.91"N)
  Upper Right ( 3754632.815, 2702950.320) (  6d22'32.10"E, 51d 2'26.55"N)
  Lower Right ( 3754632.815, 2632709.750) (  6d25'32.63"E, 50d23'15.34"N)
  Center      ( 3719425.244, 2667830.035) (  5d53' 7.40"E, 50d41'49.42"N)
  Band 1 Block=1408x1 Type=Float32, ColorInterp=Gray
    NoData Value=nan
  ```

![ET_DTM_HC_2026_FINAL_50m.tif](Create%20GeoTIFF%20sample%20data/ET_DTM_HC_2026_FINAL_50m.tif)

[tiff1.tif](Create%20GeoTIFF%20sample%20data/tiff1.tif)

- Tiff2


  ```
  Driver: GTiff/GeoTIFF
  Files: tiff2.tiff
  Size is 1114, 1035
  Coordinate System is:
  PROJCRS["ETRS89-extended / LCC Europe",
      BASEGEOGCRS["ETRS89",
          ENSEMBLE["European Terrestrial Reference System 1989 ensemble",
              MEMBER["European Terrestrial Reference Frame 1989"],
              MEMBER["European Terrestrial Reference Frame 1990"],
              MEMBER["European Terrestrial Reference Frame 1991"],
              MEMBER["European Terrestrial Reference Frame 1992"],
              MEMBER["European Terrestrial Reference Frame 1993"],
              MEMBER["European Terrestrial Reference Frame 1994"],
              MEMBER["European Terrestrial Reference Frame 1996"],
              MEMBER["European Terrestrial Reference Frame 1997"],
              MEMBER["European Terrestrial Reference Frame 2000"],
              MEMBER["European Terrestrial Reference Frame 2005"],
              MEMBER["European Terrestrial Reference Frame 2014"],
              MEMBER["European Terrestrial Reference Frame 2020"],
              ELLIPSOID["GRS 1980",6378137,298.257222101,
                  LENGTHUNIT["metre",1]],
              ENSEMBLEACCURACY[0.1]],
          PRIMEM["Greenwich",0,
              ANGLEUNIT["degree",0.0174532925199433]],
          ID["EPSG",4258]],
      CONVERSION["Europe Conformal 2001",
          METHOD["Lambert Conic Conformal (2SP)",
              ID["EPSG",9802]],
          PARAMETER["Latitude of false origin",52,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8821]],
          PARAMETER["Longitude of false origin",10,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8822]],
          PARAMETER["Latitude of 1st standard parallel",35,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8823]],
          PARAMETER["Latitude of 2nd standard parallel",65,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8824]],
          PARAMETER["Easting at false origin",4000000,
              LENGTHUNIT["metre",1],
              ID["EPSG",8826]],
          PARAMETER["Northing at false origin",2800000,
              LENGTHUNIT["metre",1],
              ID["EPSG",8827]]],
      CS[Cartesian,2],
          AXIS["northing (N)",north,
              ORDER[1],
              LENGTHUNIT["metre",1]],
          AXIS["easting (E)",east,
              ORDER[2],
              LENGTHUNIT["metre",1]],
      USAGE[
          SCOPE["Conformal mapping at scales of 1:500,000 and smaller."],
          AREA["Europe - European Union (EU) countries and candidates. Europe - onshore and offshore: Albania; Andorra; Austria; Belgium; Bosnia and Herzegovina; Bulgaria; Croatia; Cyprus; Czechia; Denmark; Estonia; Faroe Islands; Finland; France; Germany; Gibraltar; Greece; Hungary; Iceland; Ireland; Italy; Kosovo; Latvia; Liechtenstein; Lithuania; Luxembourg; Malta; Monaco; Montenegro; Netherlands; North Macedonia; Norway including Svalbard and Jan Mayen; Poland; Portugal including Madeira and Azores; Romania; San Marino; Serbia; Slovakia; Slovenia; Spain including Canary Islands; Sweden; Switzerland; Türkiye (Turkey); United Kingdom (UK) including Channel Islands and Isle of Man; Vatican City State."],
          BBOX[24.6,-35.58,84.73,44.83]],
      ID["EPSG",3034]]
  Data axis to CRS axis mapping: 2,1
  Origin = (3711183.000000000000000,2678182.140484739560634)
  Pixel Size = (15.616696588868940,-15.616696588868940)
  Metadata:
    TIFFTAG_XRESOLUTION=72
    TIFFTAG_YRESOLUTION=72
    TIFFTAG_RESOLUTIONUNIT=2 (pixels/inch)
    AREA_OR_POINT=Area
  Image Structure Metadata:
    COMPRESSION=LZW
    INTERLEAVE=PIXEL
  Corner Coordinates:
  Upper Left  ( 3711183.000, 2678182.140) (  5d45'21.85"E, 50d47'20.19"N)
  Lower Left  ( 3711183.000, 2662018.860) (  5d46'10.74"E, 50d38'19.39"N)
  Upper Right ( 3728580.000, 2678182.140) (  6d 0'40.29"E, 50d47'52.65"N)
  Lower Right ( 3728580.000, 2662018.860) (  6d 1'26.26"E, 50d38'51.75"N)
  Center      ( 3719881.500, 2670100.500) (  5d53'24.78"E, 50d43' 6.25"N)
  Band 1 Block=1114x941 Type=Byte, ColorInterp=Red
    Mask Flags: PER_DATASET ALPHA 
  Band 2 Block=1114x941 Type=Byte, ColorInterp=Green
    Mask Flags: PER_DATASET ALPHA 
  Band 3 Block=1114x941 Type=Byte, ColorInterp=Blue
    Mask Flags: PER_DATASET ALPHA 
  Band 4 Block=1114x941 Type=Byte, ColorInterp=Alpha
  ```

![tiff2.tiff](Create%20GeoTIFF%20sample%20data/tiff2.tiff)

[tiff2.tiff](Create%20GeoTIFF%20sample%20data/tiff2%201.tiff)

- Tiff3 (larger one)


  ```
  Driver: GTiff/GeoTIFF
  Files: tiff3.tiff
  Size is 12699, 11797
  Coordinate System is:
  PROJCRS["ETRS89-extended / LCC Europe",
      BASEGEOGCRS["ETRS89",
          ENSEMBLE["European Terrestrial Reference System 1989 ensemble",
              MEMBER["European Terrestrial Reference Frame 1989"],
              MEMBER["European Terrestrial Reference Frame 1990"],
              MEMBER["European Terrestrial Reference Frame 1991"],
              MEMBER["European Terrestrial Reference Frame 1992"],
              MEMBER["European Terrestrial Reference Frame 1993"],
              MEMBER["European Terrestrial Reference Frame 1994"],
              MEMBER["European Terrestrial Reference Frame 1996"],
              MEMBER["European Terrestrial Reference Frame 1997"],
              MEMBER["European Terrestrial Reference Frame 2000"],
              MEMBER["European Terrestrial Reference Frame 2005"],
              MEMBER["European Terrestrial Reference Frame 2014"],
              MEMBER["European Terrestrial Reference Frame 2020"],
              ELLIPSOID["GRS 1980",6378137,298.257222101,
                  LENGTHUNIT["metre",1]],
              ENSEMBLEACCURACY[0.1]],
          PRIMEM["Greenwich",0,
              ANGLEUNIT["degree",0.0174532925199433]],
          ID["EPSG",4258]],
      CONVERSION["Europe Conformal 2001",
          METHOD["Lambert Conic Conformal (2SP)",
              ID["EPSG",9802]],
          PARAMETER["Latitude of false origin",52,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8821]],
          PARAMETER["Longitude of false origin",10,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8822]],
          PARAMETER["Latitude of 1st standard parallel",35,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8823]],
          PARAMETER["Latitude of 2nd standard parallel",65,
              ANGLEUNIT["degree",0.0174532925199433],
              ID["EPSG",8824]],
          PARAMETER["Easting at false origin",4000000,
              LENGTHUNIT["metre",1],
              ID["EPSG",8826]],
          PARAMETER["Northing at false origin",2800000,
              LENGTHUNIT["metre",1],
              ID["EPSG",8827]]],
      CS[Cartesian,2],
          AXIS["northing (N)",north,
              ORDER[1],
              LENGTHUNIT["metre",1]],
          AXIS["easting (E)",east,
              ORDER[2],
              LENGTHUNIT["metre",1]],
      USAGE[
          SCOPE["Conformal mapping at scales of 1:500,000 and smaller."],
          AREA["Europe - European Union (EU) countries and candidates. Europe - onshore and offshore: Albania; Andorra; Austria; Belgium; Bosnia and Herzegovina; Bulgaria; Croatia; Cyprus; Czechia; Denmark; Estonia; Faroe Islands; Finland; France; Germany; Gibraltar; Greece; Hungary; Iceland; Ireland; Italy; Kosovo; Latvia; Liechtenstein; Lithuania; Luxembourg; Malta; Monaco; Montenegro; Netherlands; North Macedonia; Norway including Svalbard and Jan Mayen; Poland; Portugal including Madeira and Azores; Romania; San Marino; Serbia; Slovakia; Slovenia; Spain including Canary Islands; Sweden; Switzerland; Türkiye (Turkey); United Kingdom (UK) including Channel Islands and Isle of Man; Vatican City State."],
          BBOX[24.6,-35.58,84.73,44.83]],
      ID["EPSG",3034]]
  Data axis to CRS axis mapping: 2,1
  Origin = (3711183.000000000000000,2678181.152374202851206)
  Pixel Size = (1.369950389794472,-1.369950389794472)
  Metadata:
    TIFFTAG_XRESOLUTION=100
    TIFFTAG_YRESOLUTION=100
    TIFFTAG_RESOLUTIONUNIT=2 (pixels/inch)
    AREA_OR_POINT=Area
  Image Structure Metadata:
    COMPRESSION=LZW
    INTERLEAVE=PIXEL
  Corner Coordinates:
  Upper Left  ( 3711183.000, 2678181.152) (  5d45'21.85"E, 50d47'20.16"N)
  Lower Left  ( 3711183.000, 2662019.848) (  5d46'10.74"E, 50d38'19.42"N)
  Upper Right ( 3728580.000, 2678181.152) (  6d 0'40.29"E, 50d47'52.62"N)
  Lower Right ( 3728580.000, 2662019.848) (  6d 1'26.25"E, 50d38'51.78"N)
  Center      ( 3719881.500, 2670100.500) (  5d53'24.78"E, 50d43' 6.25"N)
  Band 1 Block=12699x82 Type=Byte, ColorInterp=Red
    Mask Flags: PER_DATASET ALPHA 
  Band 2 Block=12699x82 Type=Byte, ColorInterp=Green
    Mask Flags: PER_DATASET ALPHA 
  Band 3 Block=12699x82 Type=Byte, ColorInterp=Blue
    Mask Flags: PER_DATASET ALPHA 
  Band 4 Block=12699x82 Type=Byte, ColorInterp=Alpha
  ```

[tiff3.tiff](Create%20GeoTIFF%20sample%20data/tiff3.tiff)

#### Test Cases:

- 1

  [rasterio-rgb-byte.tif](Create%20GeoTIFF%20sample%20data/rasterio-rgb-byte.tif)

    - 3-band RGB byte imagery
    - `PixelIsArea`
    - EPSG-coded CRS
    - Uncompressed data

- 2

  [geotiff-testdata-gadas.tif](Create%20GeoTIFF%20sample%20data/geotiff-testdata-gadas.tif)

    - 4-band RGBA byte imagery

- 3

  [geotiff-testdata-geogtowgs84.tif](Create%20GeoTIFF%20sample%20data/geotiff-testdata-geogtowgs84.tif)

    - Palette raster with `ColorMap`
    - PackBits compression
    - `GeogTOWGS84GeoKey` datum transform

- 4

  [osgeo-gdal-cea.tif](Create%20GeoTIFF%20sample%20data/osgeo-gdal-cea.tif)

    - Single-band byte grayscale
    - Stripped layout
    - User-defined CRS

- 5

  [osgeo-usgs-i30dem.tif](Create%20GeoTIFF%20sample%20data/osgeo-usgs-i30dem.tif)

    - Signed 16-bit DEM
    - `PixelIsPoint`

- 6

  [geotiff-testdata-lisbon-elevation.tif](Create%20GeoTIFF%20sample%20data/geotiff-testdata-lisbon-elevation.tif)

    - Float32 DEM / scientific raster

- 7

  [geotiff-testdata-wind-direction.tif](Create%20GeoTIFF%20sample%20data/geotiff-testdata-wind-direction.tif)

    - LZW compression

- 8

  [geotiff-testdata-vanderford-cog.tif](Create%20GeoTIFF%20sample%20data/geotiff-testdata-vanderford-cog.tif)

    - Deflate compression
    - Tiled layout
    - Multiple IFDs / internal overviews

#### File Descriptions

| File                                    | Variety                                  | Key traits                                                   | CRS / georeferencing                                         | Source                                                       |
| --------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `rasterio-rgb-byte.tif`                 | RGB imagery                              | 791x718, 3 x UInt8, RGB, uncompressed, stripped              | EPSG:32618, WGS 84 / UTM zone 18N, PixelIsArea               | [https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif](https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif) |
| `geotiff-testdata-gadas.tif`            | RGBA imagery                             | 968x475, 4 x UInt8, RGB + unassociated alpha, uncompressed, stripped | user-defined Web Mercator / WGS84-style CRS text, PixelIsArea | [https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/gadas.tif](https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/gadas.tif) |
| `geotiff-testdata-geogtowgs84.tif`      | Palette + datum transform                | 101x101, UInt8 palette, PackBits, stripped, `ColorMap` present | user-defined geographic CRS with `GeogTOWGS84`transform, PixelIsArea | [https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/GeogToWGS84GeoKey5.tif](https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/GeogToWGS84GeoKey5.tif) |
| `geotiff-testdata-lisbon-elevation.tif` | Float32 elevation                        | 547x421, 1 x Float32, uncompressed, stripped                 | EPSG:4326 / WGS84, PixelIsArea                               | [https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/LisbonElevation.tif](https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/LisbonElevation.tif) |
| `osgeo-usgs-i30dem.tif`                 | Signed 16-bit DEM                        | 1479x1863, 1 x Int16, uncompressed, stripped                 | EPSG:26710, UTM Zone 10 / NAD27, PixelIsPoint                | [https://download.osgeo.org/geotiff/samples/usgs/i30dem.tif](https://download.osgeo.org/geotiff/samples/usgs/i30dem.tif) |
| `geotiff-testdata-vanderford-cog.tif`   | COG-style tiled Float32 raster           | 2581x1998 main IFD, 1 x Float32, Deflate, 512x512 tiled, 4 IFDs with internal overviews | EPSG:3031, Antarctic Polar Stereographic, PixelIsPoint       | [https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/GA4886_VanderfordGlacier_2022_EGM2008_64m-epsg3031.cog](https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/GA4886_VanderfordGlacier_2022_EGM2008_64m-epsg3031.cog) |
| `geotiff-testdata-wind-direction.tif`   | LZW Float32 scientific raster            | 232x193, 1 x Float32, LZW, stripped                          | EPSG:4326 / WGS84, PixelIsArea                               | [https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/wind_direction.tif](https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/wind_direction.tif) |
| `osgeo-gdal-cea.tif`                    | Byte grayscale + user-defined projection | 514x515, 1 x UInt8, uncompressed, stripped                   | user-defined Cylindrical Equal Area over NAD27, PixelIsArea  | [https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif](https://download.osgeo.org/geotiff/samples/gdal_eg/cea.tif) |
| `geotiff-testdata-dom1-float32.tif`     | Float32 projected raster                 | 1000x1000, 1 x Float32, uncompressed, stripped               |                                                              |                                                              |EPSG:25832, ETRS89 / UTM zone 32N, PixelIsArea[https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/dom1_32_356_5699_1_nw_2020.tif](https://raw.githubusercontent.com/GeoTIFF/test-data/main/files/dom1_32_356_5699_1_nw_2020.tif)