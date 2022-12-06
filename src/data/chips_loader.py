import fsspec
import geopandas as gpd
import os
import pandas as pd
import xarray as xr


class ChipS2:
    def __init__(self, data, df_scenes):
        self.df_scenes = df_scenes
        self.data = data

    def viz_compare_time_periods(
        self,
        bands=["B11", "B8A", "B04"],
        metrics=["p010", "p025", "p050", "p075", "p090", "pmid50", "pmid80"],
    ):

        fcc = (
            self.data[bands].sel(metric=metrics).to_array().rename({"variable": "band"})
        )
        fcc.plot.imshow(
            row="metric", col="time_period", rgb="band", robust=True, size=6
        )


class ChipsReader:
    def __init__(
        self,
        s3_uri_chips="s3://mi4people-soil-project/chips",
        cache_storage="./tmp/files/",
        check_files=False,
        secrets_file_path=None,
    ):

        # https://s3fs.readthedocs.io/en/latest/index.html#credentials

        self._s3_uri_chips = s3_uri_chips
        self._cache_storage = cache_storage
        self._check_files = check_files
        self._secrets_file_path = secrets_file_path

        # setup data
        self._uri_chip_geometries = (
            self._s3_uri_chips
            + "/sol_chem_pnts_horizons_africa_chip_geometries.parquet"
        )
        # not required anymore
        # self._uri_logs = self._s3_uri_chips + '/s2_metrics_p/s2-chips-generation.log'
        self._uri_data = self._s3_uri_chips + "/s2_metrics_p/data"
        self._uri_metadata = self._s3_uri_chips + "/s2_metrics_p/meatadata_s2_scenes"
        self._uri_data = self._s3_uri_chips + "/s2_metrics_p/data"

        # setup environmental variables with secrets if given
        if self._secrets_file_path:
            self._setup_access()

        self._fs = fsspec.filesystem(
            "filecache",
            target_protocol="s3",
            cache_storage=self._cache_storage,
            check_files=self._check_files,
        )

        # load data
        self._logs = None
        self.gemoetries = None
        self._get_geometries()

    def _get_geometries(self):
        with self._fs.open(self._uri_chip_geometries, mode="rb") as file:
            gdf = gpd.read_parquet(file)
        # self._get_logs()
        # missing_locations = []
        # for line in self._logs.split('\n'):
        #     if 'No scenes found for' in line:
        #         missing_locations.append(line.split('No scenes found for ')[1])
        # gdf['missing_data'] = False
        # gdf.loc[gdf['olc_id'].isin(missing_locations), 'missing_data'] = True
        # polygons
        self.gdf = gdf
        # same as points
        self.gdf_points = gpd.GeoDataFrame(
            self.gdf.drop("geometry", axis=1),
            geometry=gpd.points_from_xy(self.gdf.longitude, self.gdf.latitude),
            crs="epsg:4326",
        )

    # def _get_logs(self):
    #     with self._fs.open(self._uri_logs, mode='r') as file:
    #         logs = file.read()
    #     self._logs = logs

    def get_chip(self, olc_id):
        with self._fs.open(self._uri_metadata + f"/{olc_id}.csv", mode="rb") as file:
            scenes = pd.read_csv(file, index_col=0)
        with self._fs.open(self._uri_data + f"/{olc_id}.nc", mode="rb") as file:
            data = xr.open_dataset(file, engine="scipy")

        return ChipS2(data, scenes)

    def _setup_access(self):
        with open(self._secrets_file_path) as src:
            # Create connection to S3
            id, secret = src.readlines()[1].rstrip("\n").split(",")
            os.environ["AWS_ACCESS_KEY_ID"] = id
            os.environ["AWS_SECRET_ACCESS_KEY"] = secret
