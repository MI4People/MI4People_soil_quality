# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import geopandas as gpd
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile


@click.command()
@click.option("overwrite", type=bool, default=False)
def main(overwrite):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("creating sol_chem_pnts_horizons_africa.gpkg")

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    url_soilchem = "https://gitlab.com/openlandmap/compiled-ess-point-data-sets/-/raw/master/out/gpkg/sol_chem.pnts_horizons.gpkg?inline=false"
    path_soilchem = project_dir / "data/raw/target/sol_chem.pnts_horizons.gpkg"
    url_gadm = "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip"
    path_gadm = project_dir / "data/raw/aux/gadm_410-levels.zip"
    path_gadm_extracted = project_dir / "data/raw/aux/gadm_410-levels.gpkg"

    path_gadm_african_countries = (
        project_dir / "data/intermediate/aux/gadm_410-levels_africa.gpkg"
    )
    path_soilchem_africa = (
        project_dir / "data/intermediate/target/sol_chem_pnts_horizons_africa.gpkg"
    )

    if not overwrite and path_soilchem_africa.exists():
        logger.info(f"stopping script - {path_soilchem_africa} exists")
        return 0

    if not path_soilchem.exists():
        logger.info(f"downloading {url_soilchem}")
        path_soilchem.parent.mkdir(exist_ok=True, parents=True)
        open(path_soilchem, "wb").write(requests.get(url_soilchem).content)
    else:
        logger.info(f"already downloaded {url_soilchem}")

    if not path_gadm_extracted.exists():
        if not path_gadm.exists():
            logger.info(f"downloading {url_gadm}")
            path_gadm.parent.mkdir(exist_ok=True, parents=True)
            open(path_gadm, "wb").write(requests.get(url_gadm).content)
            logger.info(f"unzipping {path_gadm}")
        zipfile.ZipFile(path_gadm).extractall(path_gadm.parent)
    else:
        logger.info(f"already downloaded {url_gadm}")

    # get the iso identifiers for african countries from the lowres countries
    # we do not directly use the lowres dataset for country polygons because of the low resolution
    # this does not contain smaller islands and also misses other points close to the coast in the spatial join later
    logger.info(f"reading naturalearth_lowres")
    countries_lowres = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    iso_a3_african_countries = countries_lowres.query('continent == "Africa"')[
        "iso_a3"
    ].values

    logger.info(f"reading {path_gadm_extracted}")
    countries = gpd.read_file(path_gadm_extracted, layer="ADM_0")
    logger.info(f"subsetting gadm to african countries")
    african_countries = countries.query(f"GID_0.isin({list(iso_a3_african_countries)})")
    if not path_gadm_african_countries.exists():
        logger.info(f"writing {path_gadm_african_countries}")
        path_gadm_african_countries.parent.mkdir(exist_ok=True, parents=True)
        african_countries.to_file(path_gadm_african_countries, driver="GPKG")

    logger.info(f"reading {path_soilchem}")
    soilchem = gpd.read_file(path_soilchem)
    logger.info(
        f"creat african subset of soilchem by inner spatial join of african countries gadm and soilchem"
    )
    soilchem_africa = gpd.sjoin(soilchem, african_countries, how="inner")

    # Note: If we try to write the file with get the error `ValueError: Invalid field type <class 'bytes'>`.
    # Identify rows and columns causing the issue and convert them to str
    for col in soilchem_africa.columns.drop("geometry"):
        is_byte = soilchem_africa[col].apply(lambda x: type(x)) == bytes
        count_byte_entries = is_byte.sum()
        if count_byte_entries > 0:
            logger.warn(
                f"column {col} contains {count_byte_entries} byte entries: {soilchem_africa.loc[is_byte, col].values}"
            )
            logger.warn(f"converting column {col} with byte entries to str")
            soilchem_africa[col] = soilchem_africa[col].astype(str)

    if not path_soilchem_africa.exists():
        logger.info(f"writing {path_soilchem_africa}")
        path_soilchem_africa.parent.mkdir(exist_ok=True, parents=True)
        soilchem_africa.to_file(path_soilchem_africa, driver="GPKG")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
