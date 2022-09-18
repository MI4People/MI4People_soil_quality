# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import geoimgchips
import geopandas as gpd
from pathlib import Path


@click.command()
@click.argument("overwrite", type=bool, default=False)
def main(overwrite):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("creating sol_chem_pnts_horizons_africa_chip_geometries.gpkg")

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    path_soilchem_africa = (
        project_dir / "data/intermediate/target/sol_chem_pnts_horizons_africa.gpkg"
    )
    path_soilchem_africa_chip_geometries = (
        project_dir
        / "data/intermediate/aux/sol_chem_pnts_horizons_africa_chip_geometries.gpkg"
    )

    if not overwrite and path_soilchem_africa_chip_geometries.exists():
        logger.info(f"stopping script - {path_soilchem_africa_chip_geometries} exists")
        return 0

    soilchem_africa = gpd.read_file(path_soilchem_africa)

    # size of the chip
    # e.g. 1280 => 256 x 256 10m pixels, size can differ if res != 10
    buffer = (2**7) * 10 / 2

    # aligns chip borders for a given pixel resolution
    res = 10

    # subset of columns from the source file to keep for the destination files
    # default `None` will keep all
    # if given, geometry needs to be included
    src_keep_columns = ["olc_id", "geometry"]

    # overwrite destination files if they exist
    overwrite = True

    path_soilchem_africa_chip_geometries = Path(path_soilchem_africa_chip_geometries)

    points = gpd.read_file(filename=path_soilchem_africa)
    if src_keep_columns is not None:
        points = points[src_keep_columns]
    points.head()

    assert "latitude" not in points.columns
    assert "longitude" not in points.columns
    assert "chip_id" not in points.columns

    points["latitude"] = points.geometry.y
    points["longitude"] = points.geometry.x
    points = points.assign(chip_id=range(1, points.shape[0] + 1))
    points.head()

    logger.info(f"creating chips")
    chips = geoimgchips.s2_raster_aligned_chips_from_points(
        points, buffer=buffer, res=res
    )

    logger.info(f"writing {path_soilchem_africa_chip_geometries}")
    Path(path_soilchem_africa_chip_geometries).parent.mkdir(exist_ok=True)
    chips.to_file(path_soilchem_africa_chip_geometries)

    return 0


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
