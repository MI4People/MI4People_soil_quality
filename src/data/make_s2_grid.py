# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

import geopandas as gpd
from pathlib import Path
import requests
import fiona

fiona.drvsupport.supported_drivers[
    "LIBKML"
] = "rw"  # enable KML support which is disabled by default


@click.command()
# @click.option("overwrite", type=bool, default=False)
def main():  # overwrite
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    overwrite = False
    logger = logging.getLogger(__name__)
    logger.info("creating s2_grid_africa.gpkg")

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    url_kml = "https://sentinel.esa.int/documents/247904/1955685/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"
    path_kml = (
        project_dir
        / "data/raw/aux/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"
    )
    path_s2_grid = project_dir / "data/intermediate/aux/s2_tile_grid.gpkg"
    path_s2_grid_africa = project_dir / "data/intermediate/aux/s2_tile_grid_africa.gpkg"
    path_gadm_africa = project_dir / "data/intermediate/aux/gadm_410-levels_africa.gpkg"

    if not overwrite and path_s2_grid_africa.exists():
        logger.info(f"stopping script - {path_s2_grid_africa} exists")
        return 0

    if not path_kml.exists() or overwrite:
        logger.info(f"downloading {url_kml}")
        path_kml.parent.mkdir(exist_ok=True, parents=True)
        open(path_kml, "wb").write(requests.get(url_kml).content)
    else:
        logger.info(f"already downloaded {url_kml}")

    if not path_s2_grid_africa.exists() or overwrite:

        if not path_s2_grid.exists() or overwrite:
            logger.info(f"reading {path_kml}")
            s2_tiles = gpd.read_file(path_kml)
            logger.info(f"extracting polygon")
            s2_tiles = s2_tiles.rename({"geometry": "geometry_collection"}, axis=1)
            s2_tiles["geometry"] = s2_tiles["geometry_collection"].apply(
                lambda x: x.geoms[0]
            )
            logger.info(f"Subsetting to geometry and Name fields")
            s2_tiles = s2_tiles[["geometry", "Name"]]
            logger.info(f"writing {path_s2_grid}")
            s2_tiles.to_file(path_s2_grid, driver="GPKG")
        else:
            logger.info(f"reading {path_s2_grid}")
            s2_tiles = gpd.read_file(path_s2_grid)
            logger.info(f"writing {path_s2_grid}")

        logger.info(f"reading {path_gadm_africa}")
        gdf_africa = gpd.read_file(path_gadm_africa)

        logger.info("subsetting s2 tile grid to Africa")
        s2_tiles_africa = (
            gpd.sjoin(s2_tiles, gdf_africa[["geometry"]], how="inner")
            .drop("index_right", axis=1)
            .drop_duplicates()
        )
        logger.info(f"writing {path_s2_grid_africa}")
        s2_tiles_africa.to_file(path_s2_grid_africa, driver="GPKG")
    else:
        logger.info(f"already exists {path_s2_grid_africa}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
