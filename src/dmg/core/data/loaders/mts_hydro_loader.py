import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pydantic import BaseModel, ConfigDict

from dmg.core.data.loaders.base import BaseLoader
from dmg.core.utils import PathWeightedAgg, reachability_matrix
from dmg.core.utils.pydantic_compat import PYDANTIC_V2

log = logging.getLogger(__name__)


class MtsHydroLoader(BaseLoader):
    """Data loader for multi-timescale (MTS) hydrological data.

    Loads distributed hourly forcing, attribute, runoff, and topology data
    for multi-timescale differentiable model experiments. Data is streamed
    in year-sized chunks to support large datasets that do not fit in memory.

    Parameters
    ----------
    config
        Configuration dictionary containing observation paths, model settings,
        and train/valid/test time ranges.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config)

        observation_paths = config['observations']['observation']
        path_forcing = Path(observation_paths['path_forcing'])
        path_attrs = Path(observation_paths['path_attrs'])
        path_topo = Path(observation_paths['path_topo'])
        path_runoff = Path(observation_paths['path_runoff'])
        path_gauges = Path(observation_paths['path_gauges'])
        path_units = Path(observation_paths['path_units'])
        runoff_start_time = observation_paths['runoff_start_time']
        preprocessing_paths = config['observations']['preprocessing']

        forcing_order = config['delta_model']['nn_model']['high_freq_model']['forcings']
        attribute_order = config['delta_model']['nn_model']['high_freq_model'][
            'attributes'
        ]
        routing_attr_order = config['delta_model']['nn_model']['high_freq_model'][
            'attributes2'
        ]
        train_start_year = pd.to_datetime(config['train']['start_time']).year
        train_end_year = pd.to_datetime(config['train']['end_time']).year
        valid_start_year = pd.to_datetime(config['valid']['start_time']).year
        valid_end_year = pd.to_datetime(config['valid']['end_time']).year
        test_start_year = pd.to_datetime(config['test']['start_time']).year
        test_end_year = pd.to_datetime(config['test']['end_time']).year
        warmup_days = config['delta_model']['phy_model']['low_freq_model'][
            'window_size'
        ]
        chunk_year_size = config['train']['chunk_year_size']

        self.preprocessing_paths = preprocessing_paths
        self.load_norm_stats()
        stats_dict = self.stats_dict
        self.data_reader = DistributedDataReader(
            path_forcing=path_forcing,
            path_attrs=path_attrs,
            path_topo=path_topo,
            path_runoff=path_runoff,
            path_gauges=path_gauges,
            path_units=path_units,
            runoff_thres=stats_dict['quantile'],
            runoff_start_time=runoff_start_time,
            forcing_order=forcing_order,
            attribute_order=attribute_order,
            routing_attr_order=routing_attr_order,
            chunk_year_size=chunk_year_size,
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            valid_start_year=valid_start_year,
            valid_end_year=valid_end_year,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
            warmup_days=warmup_days,
            selected_gauges=stats_dict['gauge_ids'],
            selected_basins=stats_dict['unit_ids'],
        )

        # current loaded dataset (train/valid/test)
        self.dataset = None

        # saved datasets for single-chunk training/validation/testing
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        # normalization arrays
        self.norm_stats = {'stdarray': np.array(stats_dict['stds'])}

    def load_dataset(self, mode: str = None):
        """Load dataset for the specified mode into self.dataset.

        If the number of chunks is 1, loads the entire dataset into memory.
        Otherwise, sets self.dataset to a generator that yields chunks.

        Parameters
        ----------
        mode
            One of 'train', 'valid', 'test', or 'simulation'.

        Raises
        ------
        ValueError
            If mode is not one of the accepted values.
        """
        if mode == 'train':
            if self.train_dataset is not None:
                self.dataset = self.train_dataset
            else:
                dataset_generator = self.data_reader.yield_train_set()
                if self.data_reader.num_train_chunks == 1:
                    self.train_dataset = list(dataset_generator)
                    self.dataset = self.train_dataset
                else:
                    self.dataset = dataset_generator
        elif mode == 'valid':
            if self.valid_dataset is not None:
                self.dataset = self.valid_dataset
            else:
                dataset_generator = self.data_reader.yield_valid_set()
                if self.data_reader.num_valid_chunks == 1:
                    self.valid_dataset = list(dataset_generator)
                    self.dataset = self.valid_dataset
                else:
                    self.dataset = dataset_generator
        elif mode in ['test', 'simulation']:
            if self.test_dataset is not None:
                self.dataset = self.test_dataset
            else:
                dataset_generator = self.data_reader.yield_test_set()
                if self.data_reader.num_test_chunks == 1:
                    self.test_dataset = list(dataset_generator)
                    self.dataset = self.test_dataset
                else:
                    self.dataset = dataset_generator
        else:
            raise ValueError(
                "mode should be one of ['train', 'valid', 'test', 'simulation']",
            )

    def get_dataset(self):
        """Yield preprocessed data from the loaded dataset."""
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Please call load_dataset() first.")
        for data in self.dataset:
            yield self.preprocessor.transform(data)

    def _preprocess_data(self) -> dict[str, torch.Tensor]:
        """Read, preprocess, and return data as dictionary of torch tensors."""
        log.info("Preprocessing data...")

    def load_norm_stats(self) -> None:
        """Load normalization statistics from preprocessing paths.

        Raises
        ------
        FileNotFoundError
            If the stats or preprocessor file does not exist.
        json.JSONDecodeError
            If the stats file cannot be parsed as JSON.
        """
        try:
            preprocessing_paths = self.config['observations']['preprocessing']
            with open(Path(preprocessing_paths['path_stats']), 'rb') as f:
                self.stats_dict = json.load(f)
            self.preprocessor = DistributedDataPreprocessor()
            self.preprocessor.load_stat(Path(preprocessing_paths['path_preprocess']))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise type(e)(f"Error loading normalization statistics: {e}") from e

    def cleanup_memory(self) -> None:
        """Clean up loaded datasets to free memory."""
        self.dataset = None


class DistributedDataSchema(BaseModel):
    """Pydantic schema for a single MTS data chunk.

    Holds all tensors required for one training/validation/test chunk,
    including dynamic inputs, static attributes, runoff targets, and
    topology/routing information.
    """

    # target: TensorType["n_gages", "t"]
    # dyn_input: TensorType["n_units", "t", "d"]
    # static_input: TensorType["n_units", "s"]
    # rout_static_input: Optional[TensorType["n_gages", "n_units", "rs"]]

    # ac_all: TensorType["n_units"]
    # elev_all: TensorType["n_units"]
    # areas: TensorType["n_units"]

    # time: TensorType["t"]
    # topo: TensorType["n_gages", "n_units"]
    # unit: list[int]  # n_units
    # gauge: list[str]  # n_gages, id
    # gauge_index: TensorType["n_gages"]  # idx for gage-wise normalized loss

    # scaled_target: Optional[TensorType["n_gages", "t"]] = None
    # scaled_dyn_input: Optional[TensorType["n_units", "window_size", "d"]] = None
    # scaled_static_input: Optional[TensorType["n_units", "s"]] = None
    # scaled_rout_static_input: Optional[TensorType["n_gages", "n_units", "rs"]] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:

        class Config:
            """Pydantic configuration."""

            arbitrary_types_allowed = True


class DistributedDataReader:
    """Reads and chunks distributed hourly MTS forcing and runoff data.

    Parameters
    ----------
    path_forcing
        Path to the directory containing yearly forcing NetCDF files.
    path_attrs
        Path to the basin attributes NetCDF file.
    path_topo
        Path to the topology JSON file describing the river network.
    path_runoff
        Path to the runoff (streamflow) NetCDF file.
    path_gauges
        Path to the gauge metadata CSV file.
    path_units
        Path to the catchment divide GeoPackage file.
    runoff_start_time
        Start datetime string for the runoff time series (e.g. '2010-01-01').
    forcing_order
        Ordered list of forcing variable names (e.g. ['P', 'Temp', 'PET']).
    attribute_order
        Ordered list of basin attribute names.
    routing_attr_order
        Ordered list of routing attribute names.
    chunk_year_size
        Number of years per data chunk. Default is 1.
    warmup_days
        Number of warmup days prepended to each chunk. Default is 365.
    runoff_thres
        Per-gauge runoff threshold below which values are set to NaN.
    train_start_year
        First year of the training period.
    train_end_year
        Last year of the training period.
    valid_start_year
        First year of the validation period.
    valid_end_year
        Last year of the validation period.
    test_start_year
        First year of the test period.
    test_end_year
        Last year of the test period.
    selected_gauges
        Subset of gauge IDs to use. If None, all available gauges are used.
    selected_basins
        Subset of basin IDs to use. If None, all available basins are used.
    """

    def __init__(
        self,
        path_forcing: Union[str, Path],
        path_attrs: Union[str, Path],
        path_topo: Union[str, Path],
        path_runoff: Union[str, Path],
        path_gauges: Union[str, Path],
        path_units: Union[str, Path],
        runoff_start_time: str,
        forcing_order: list[str],
        attribute_order: list[str],
        routing_attr_order: list[str],
        chunk_year_size: int = 1,
        warmup_days: int = 365,
        runoff_thres: list[float] = None,
        train_start_year: int = None,
        train_end_year: int = None,
        valid_start_year: int = None,
        valid_end_year: int = None,
        test_start_year: int = None,
        test_end_year: int = None,
        selected_gauges: list[str] = None,
        selected_basins: list[int] = None,
    ):
        self.path_forcing = path_forcing
        self.path_attrs = path_attrs
        self.path_topo = path_topo
        self.path_runoff = path_runoff
        self.path_gauges = path_gauges
        self.path_units = path_units
        self.runoff_thres = runoff_thres
        self.runoff_start_time = runoff_start_time
        self.forcing_order = forcing_order
        self.attribute_order = attribute_order
        self.routing_attr_order = routing_attr_order

        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.valid_start_year = valid_start_year
        self.valid_end_year = valid_end_year
        self.test_start_year = test_start_year
        self.test_end_year = test_end_year
        self.chunk_year_size = chunk_year_size
        self.warmup_days = warmup_days
        self.selected_gauges = selected_gauges
        self.selected_basins = selected_basins

        self.num_train_chunks = (
            (
                (self.train_end_year - self.train_start_year + 1)
                + self.chunk_year_size
                - 1
            )
            // self.chunk_year_size
            if self.train_start_year is not None and self.train_end_year is not None
            else 0
        )
        self.num_valid_chunks = (
            (
                (self.valid_end_year - self.valid_start_year + 1)
                + self.chunk_year_size
                - 1
            )
            // self.chunk_year_size
            if self.valid_start_year is not None and self.valid_end_year is not None
            else 0
        )
        self.num_test_chunks = (
            ((self.test_end_year - self.test_start_year + 1) + self.chunk_year_size - 1)
            // self.chunk_year_size
            if self.test_start_year is not None and self.test_end_year is not None
            else 0
        )

    @staticmethod
    def _get_element_ids(
        path_forcing: Union[str, Path],
        path_attrs: Union[str, Path],
        path_topo: Union[str, Path],
        path_runoff: Union[str, Path],
        path_gauges: Union[str, Path],
        runoff_start_time: str,
        area_thres: float,
        years: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine the valid gauge and basin IDs after applying data filters.

        Intersects forcing, attribute, runoff, and topology data sources to
        find the set of gauges and basins that have sufficient data coverage.
        Gauges are filtered by drainage area and require at least 80% of their
        upstream basins to have forcing and attribute data.

        Parameters
        ----------
        path_forcing
            Path to the directory containing yearly forcing NetCDF files.
        path_attrs
            Path to the basin attributes NetCDF file.
        path_topo
            Path to the topology JSON file describing the river network.
        path_runoff
            Path to the runoff NetCDF file.
        path_gauges
            Path to the gauge metadata CSV file.
        runoff_start_time
            Start datetime string for the runoff time series.
        area_thres
            Maximum drainage area (km²) for gauge inclusion.
        years
            List of years used to check data availability.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Arrays of selected gauge IDs and selected basin IDs.
        """
        # forcing basins
        xr_forcing = xr.open_dataset(f'{path_forcing}/forcing_{years[0]}.nc')
        basin_forcing = xr_forcing['gauge'].data
        xr_forcing.close()

        # attributes basins
        xr_attrs = xr.open_dataset(path_attrs)
        basin_attrs = pd.to_numeric(
            pd.Series(xr_attrs['gage'].data).str.replace('cat-', '', regex=False),
            errors='coerce',
        ).astype(int)
        attrs_duplicated_indexes = (
            pd.Series(basin_attrs).drop_duplicates(keep='first').index.values
        )
        basin_attrs = basin_attrs[attrs_duplicated_indexes]
        xr_attrs.close()

        # runoff gauges
        xr_runoff = xr.open_dataset(path_runoff)
        runoff_times = pd.date_range(
            start=runoff_start_time,
            periods=xr_runoff['time'].shape[0],
            freq='h',
        )
        runoff_time_indexes = np.where(runoff_times.year.isin(years))[0]
        gauge_runoff = xr_runoff['gauge'].data
        runoff = xr_runoff['runoff'][:, runoff_time_indexes].data
        mask = ~np.isnan(runoff).all(axis=1)
        gauge_runoff = gauge_runoff[mask]
        xr_runoff.close()

        # topology gauges and basins
        with open(path_topo) as f:
            gage_topo = json.load(f)
        G = nx.DiGraph()
        G.add_nodes_from(gage_topo['nodes'])
        G.add_edges_from(gage_topo['edges'])
        gauge_hf_dict = {}
        for gid, uid in gage_topo['gage_hf'].items():
            ancestors = nx.ancestors(G, uid)
            ancestors.add(int(uid))
            gauge_hf_dict[gid] = ancestors
        df_topo = pd.concat(
            [
                pd.DataFrame({'gauge': key, 'unit': list(value)})
                for key, value in gauge_hf_dict.items()
            ],
            ignore_index=True,
        )

        # gauge info
        gauge_info = pd.read_csv(Path(path_gauges))
        gauge_info['gauge_id'] = gauge_info['STAID'].astype(str).str.zfill(8)

        # filter gauges and basins
        df_topo = df_topo[
            df_topo['gauge'].isin(
                gauge_info.loc[gauge_info['DRAIN_SQKM'] < area_thres, 'gauge_id'],
            )
        ].reset_index(drop=True)
        df_topo = df_topo[df_topo['gauge'].isin(gauge_runoff)].reset_index(drop=True)
        df_topo['has_data'] = 0
        basins = np.intersect1d(basin_forcing, basin_attrs)
        df_topo.loc[df_topo['unit'].isin(basins), 'has_data'] = 1
        avail_unit_ratio = df_topo.groupby('gauge')['has_data'].mean()
        gauges = avail_unit_ratio[avail_unit_ratio >= 0.8].index.values
        df_topo = df_topo[
            df_topo['gauge'].isin(gauges) & df_topo['unit'].isin(basins)
        ].reset_index(drop=True)
        df_map = pd.DataFrame(
            {
                'gauge': gage_topo['gage_hf'].keys(),
                'unit': gage_topo['gage_hf'].values(),
            },
        )
        missing_gauges = df_map.loc[
            df_map['unit'].isin(set(df_map['unit']) - set(df_topo['unit'])),
            'gauge',
        ]
        df_topo = df_topo[~df_topo['gauge'].isin(missing_gauges)].reset_index(drop=True)
        df_topo = df_topo.sort_values(by=['gauge', 'unit']).reset_index(drop=True)
        df_topo['is_upstream'] = 1
        df_topo = df_topo.pivot_table(
            index='gauge',
            columns='unit',
            values='is_upstream',
            fill_value=0,
        )
        selected_gauges = df_topo.index.values
        selected_basins = df_topo.columns.values

        return selected_gauges, selected_basins

    def get_element_ids(
        self, area_thres: float, years: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get selected gauge and basin IDs based on area threshold and years.

        Parameters
        ----------
        area_thres
            Maximum drainage area (km²) for gauge inclusion.
        years
            List of years used to check data availability.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Arrays of selected gauge IDs and selected basin IDs.
        """
        return self._get_element_ids(
            path_forcing=self.path_forcing,
            path_attrs=self.path_attrs,
            path_topo=self.path_topo,
            path_runoff=self.path_runoff,
            path_gauges=self.path_gauges,
            runoff_start_time=self.runoff_start_time,
            area_thres=area_thres,
            years=years,
        )

    @staticmethod
    def _read_distributed_hourly_data(
        path_forcing: Union[str, Path],
        path_attrs: Union[str, Path],
        path_topo: Union[str, Path],
        path_runoff: Union[str, Path],
        path_units: Union[str, Path],
        runoff_start_time: str,
        forcing_order: list[str],
        attribute_order: list[str],
        routing_attr_order: list[str],
        years: list[int],
        selected_gauges: list[str],
        selected_basins: list[int],
        runoff_thres: list[float],
    ) -> DistributedDataSchema:
        """Read and assemble a distributed hourly data chunk for the given years.

        Loads forcing (P, Temp, PET), basin attributes, runoff targets, and
        topology/routing data. Aligns all data sources to the selected gauges
        and basins, computes topological routing attributes, and returns
        everything as a typed schema.

        Parameters
        ----------
        path_forcing
            Path to the directory containing yearly forcing NetCDF files.
        path_attrs
            Path to the basin attributes NetCDF file.
        path_topo
            Path to the topology JSON file describing the river network.
        path_runoff
            Path to the runoff NetCDF file.
        path_units
            Path to the catchment divide GeoPackage file.
        runoff_start_time
            Start datetime string for the runoff time series.
        forcing_order
            Ordered list of forcing variable names.
        attribute_order
            Ordered list of basin attribute names.
        routing_attr_order
            Ordered list of routing attribute names.
        years
            List of years to read (including any warmup years).
        selected_gauges
            Gauge IDs to include.
        selected_basins
            Basin IDs to include.
        runoff_thres
            Per-gauge runoff thresholds; values below threshold are set to NaN.

        Returns
        -------
        DistributedDataSchema
            Schema containing all tensors for the requested years.
        """

        def get_element_indexes(
            element_array: np.ndarray,
            elements: Union[np.ndarray, list],
        ) -> np.ndarray:
            df = pd.DataFrame(
                {'element': element_array, 'local_ind': np.arange(len(element_array))},
            )
            df = df.merge(
                pd.DataFrame(
                    {'element': elements, 'global_ind': np.arange(len(elements))},
                ),
            )
            return df.sort_values(by='global_ind')['local_ind'].values

        # read forcing data
        P = []
        Temp = []
        PET = []
        basin_forcing = None
        for year in years:
            xr_forcing = xr.open_dataset(f'{path_forcing}/forcing_{year}.nc')
            if basin_forcing is None:
                basin_forcing = xr_forcing['gauge'].data
            P.append(xr_forcing['P'].data)
            Temp.append(xr_forcing['T'].data)
            PET.append(xr_forcing['PET'].data)
            xr_forcing.close()
        P = np.concatenate(P, axis=1)
        Temp = np.concatenate(Temp, axis=1)
        PET = np.concatenate(PET, axis=1)

        # read attributes data
        xr_attrs = xr.open_dataset(path_attrs)
        basin_attrs = (
            pd.Series(xr_attrs['gage'].data)
            .str.extract(r'(\d+)$')[0]
            .values.astype(int)
        )
        attrs_duplicated_indexes = (
            pd.Series(basin_attrs).drop_duplicates(keep='first').index.values
        )
        basin_attrs = basin_attrs[attrs_duplicated_indexes]
        attrs_indexes = np.array(
            [xr_attrs['attr'].data.tolist().index(key) for key in attribute_order],
        )
        attr_names = xr_attrs['attr'].data[attrs_indexes]
        attrs = xr_attrs['__xarray_dataarray_variable__'].data[attrs_indexes, :][
            :,
            attrs_duplicated_indexes,
        ]
        xr_attrs.close()

        # read runoff data
        xr_runoff = xr.open_dataset(path_runoff)
        runoff_times = pd.date_range(
            start=runoff_start_time,
            periods=xr_runoff['time'].shape[0],
            freq='h',
        )
        runoff_time_indexes = np.where(runoff_times.year.isin(years))[0]
        gauge_runoff = xr_runoff['gauge'].data
        runoff = xr_runoff['runoff'][:, runoff_time_indexes].data
        xr_runoff.close()

        # align forcing and runoff time
        end_runoff_time = runoff_times[runoff_times.year.isin(years)][-1]
        if end_runoff_time.month == 12 or end_runoff_time.day == 31:
            # incomplete runoff data, pad with NaN
            runoff = np.concatenate(
                [
                    np.full((runoff.shape[0], P.shape[1] - runoff.shape[1]), np.nan),
                    runoff,
                ],
                axis=1,
            )
        else:
            # truncate data
            min_len = min(P.shape[1], runoff.shape[1])
            P = P[:, :min_len]
            Temp = Temp[:, :min_len]
            PET = PET[:, :min_len]
            runoff = runoff[:, :min_len]
        times = (
            pd.date_range(
                start=f'{years[0]}-01-01',
                periods=runoff.shape[1],
                freq='h',
            ).astype(int)
            // 10**9
        )

        # read topology data
        with open(path_topo) as f:
            gage_topo = json.load(f)
        G = nx.DiGraph()
        G.add_nodes_from(gage_topo['nodes'])
        G.add_edges_from(gage_topo['edges'])
        subG = G.subgraph(selected_basins)
        # selected_basins = [int(node) for node in nx.topological_sort(subG)]
        outlets = [gage_topo['gage_hf'][gauge] for gauge in selected_gauges]
        topo = reachability_matrix(subG, outlets, selected_basins)

        # read unit metadata
        divides = gpd.read_file(path_units, layer="divides")

        # indexing by sorted ids
        forcing_basin_indexes = get_element_indexes(basin_forcing, selected_basins)
        P_c = P[forcing_basin_indexes, :].clip(min=0.0)  # safeguard against data errors
        Temp_c = Temp[forcing_basin_indexes, :]
        PET_c = PET[forcing_basin_indexes, :].clip(
            min=0.0,
        )  # safeguard against data errors
        data_map = {'P': P_c, 'Temp': Temp_c, 'PET': PET_c}
        dyn_input = np.zeros(P_c.shape + (3,), dtype=np.float32)
        for i in range(3):
            dyn_input[:, :, i] = data_map[forcing_order[i]]

        runoff_gauge_indexes = get_element_indexes(gauge_runoff, selected_gauges)
        target = runoff[runoff_gauge_indexes, :]
        if runoff_thres is not None:
            target[target < np.array(runoff_thres)[:, None]] = np.nan
        attrs_basin_indexes = get_element_indexes(basin_attrs, selected_basins)
        static_input = attrs[:, attrs_basin_indexes].T
        elev_all = static_input[:, attribute_order.index('meanelevation')]
        divide_basin_indexes = get_element_indexes(
            divides['divide_id'],
            'cat-' + pd.Series(selected_basins).astype(str),
        )
        areas = divides.loc[divide_basin_indexes]['areasqkm'].values
        ac_all = divides.loc[divide_basin_indexes]['tot_drainage_areasqkm'].values
        lengths = divides.loc[divide_basin_indexes]['lengthkm'].values

        # topological aggregated attributes
        attr_dict = {
            node: {key: float(value[i]) for i, key in enumerate(attr_names)}
            for node, value in zip(selected_basins, static_input)
        }
        for i, node in enumerate(selected_basins):
            attr_dict[node]['areasqkm'] = float(areas[i])
            attr_dict[node]['lengthkm'] = float(lengths[i])
        nx.set_node_attributes(subG, attr_dict)
        pairs = [
            (selected_basins[j], outlets[i])
            for i in range(topo.shape[0])
            for j in range(topo.shape[1])
            if topo[i, j] == 1
        ]
        rout_static_input = []
        for attr in routing_attr_order:
            if attr not in ['lengthkm', 'catchsize']:
                pwm = PathWeightedAgg(subG, x_attr=attr, y_attr="areasqkm")
                agg_out = pwm.query_many(pairs, reduction='mean')
            else:
                pwm = PathWeightedAgg(subG, x_attr=attr, y_attr=None)
                agg_out = pwm.query_many(pairs, reduction='sum')
            agg_out_mat = np.full(topo.shape, np.nan)
            for k, i, j in zip(range(len(pairs)), *np.where(topo == 1)):
                agg_out_mat[i, j] = agg_out[k]
            rout_static_input.append(agg_out_mat)
        rout_static_input = np.stack(rout_static_input, axis=-1)

        # to torch tensor
        dyn_input = torch.tensor(dyn_input, dtype=torch.float32)
        static_input = torch.tensor(static_input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        rout_static_input = torch.tensor(rout_static_input, dtype=torch.float32)
        areas = torch.tensor(areas, dtype=torch.float32)
        ac_all = torch.tensor(ac_all, dtype=torch.float32)
        elev_all = torch.tensor(elev_all, dtype=torch.float32)
        topo = torch.tensor(topo, dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.int32)
        gauge_index = torch.tensor(np.arange(len(selected_gauges)), dtype=torch.int32)

        return DistributedDataSchema(
            target=target,
            dyn_input=dyn_input,
            static_input=static_input,
            rout_static_input=rout_static_input,
            areas=areas,
            ac_all=ac_all,
            elev_all=elev_all,
            unit=selected_basins,
            time=times,
            gauge=selected_gauges,
            gauge_index=gauge_index,
            topo=topo,
        )

    def read_distributed_hourly_data(self, years: list[int]) -> DistributedDataSchema:
        """Read distributed hourly data for the specified years.

        Parameters
        ----------
        years
            List of calendar years to include in the returned chunk.

        Returns
        -------
        DistributedDataSchema
            Schema containing all tensors for the requested years.
        """
        return self._read_distributed_hourly_data(
            path_forcing=self.path_forcing,
            path_attrs=self.path_attrs,
            path_topo=self.path_topo,
            path_runoff=self.path_runoff,
            path_units=self.path_units,
            runoff_start_time=self.runoff_start_time,
            forcing_order=self.forcing_order,
            attribute_order=self.attribute_order,
            routing_attr_order=self.routing_attr_order,
            years=years,
            selected_gauges=self.selected_gauges,
            selected_basins=self.selected_basins,
            runoff_thres=self.runoff_thres,
        )

    def yield_chunk_set(self, start_year: int, end_year: int, shuffle: bool = False):
        """Yield data chunks spanning start_year to end_year, each prepended with warmup days.

        Parameters
        ----------
        start_year
            First calendar year of the period to yield.
        end_year
            Last calendar year of the period to yield (inclusive).
        shuffle
            If True, yield chunks in random order. Default is False.

        Yields
        ------
        DistributedDataSchema
            One chunk of data per call, covering ``chunk_year_size`` years plus
            ``warmup_days`` of preceding data.
        """
        chunk_starts = list(range(start_year, end_year + 1, self.chunk_year_size))
        if shuffle:
            perm = torch.randperm(len(chunk_starts)).tolist()
            chunk_starts = [chunk_starts[i] for i in perm]
        for i in chunk_starts:
            years = list(range(i, min(i + self.chunk_year_size, end_year + 1)))
            warmup_dates = pd.date_range(
                end=f'{years[0]}-01-01',
                periods=self.warmup_days + 1,
                freq='d',
            )[:-1]
            warm_years = warmup_dates.year.unique().tolist()
            years = warm_years + years
            data = self.read_distributed_hourly_data(years)
            pre_read_dates = pd.date_range(
                start=f'{warm_years[0]}-01-01',
                end=f'{warm_years[-1]}-12-31',
                freq='d',
            )
            start_time_index = (len(pre_read_dates) - len(warmup_dates)) * 24
            data.target = data.target[:, start_time_index:]
            data.dyn_input = data.dyn_input[:, start_time_index:, :]
            data.time = data.time[start_time_index:]
            yield data

    def yield_train_set(self):
        """Yield training chunks in random order.

        Raises
        ------
        ValueError
            If train start/end years were not provided at construction.
        """
        if self.train_start_year is None or self.train_end_year is None:
            raise ValueError("train years are not specified.")
        yield from self.yield_chunk_set(
            self.train_start_year,
            self.train_end_year,
            shuffle=True,
        )

    def yield_valid_set(self):
        """Yield validation chunks in chronological order.

        Raises
        ------
        ValueError
            If valid start/end years were not provided at construction.
        """
        if self.valid_start_year is None or self.valid_end_year is None:
            raise ValueError("validation years are not specified.")
        yield from self.yield_chunk_set(
            self.valid_start_year,
            self.valid_end_year,
            shuffle=False,
        )

    def yield_test_set(self):
        """Yield test chunks in chronological order.

        Raises
        ------
        ValueError
            If test start/end years were not provided at construction.
        """
        if self.test_start_year is None or self.test_end_year is None:
            raise ValueError("test years are not specified.")
        yield from self.yield_chunk_set(
            self.test_start_year,
            self.test_end_year,
            shuffle=False,
        )


class DistributedDataPreprocessor:
    """Normalizer and transformer for MTS distributed data chunks.

    Fits mean/std statistics over a ``DistributedDataSchema`` and applies
    z-score normalization with optional log-transforms for precipitation-like
    dynamic inputs and streamflow targets.

    Parameters
    ----------
    norm_dyn_indexes
        Indices of dynamic input channels to log-transform before normalizing.
        If None or empty, no log-transform is applied.
    use_norm_target
        If True, apply a log-transform to the runoff target before normalizing.
    """

    def __init__(
        self,
        norm_dyn_indexes: list[int] = None,
        use_norm_target: bool = False,
    ):
        self.mean = {}
        self.std = {}
        self.norm_dyn_indexes = norm_dyn_indexes
        self.use_norm_target = use_norm_target

    @staticmethod
    def _nanstd(
        x: torch.Tensor,
        dim: Union[int, list],
        keepdim: bool = False,
        unbiased: bool = True,
    ) -> torch.Tensor:
        mask = ~torch.isnan(x)
        count = mask.sum(dim=dim, keepdim=keepdim)

        mean = torch.nanmean(x, dim=dim, keepdim=True)
        sq_diff = (x - mean) ** 2
        sq_diff[~mask] = 0  # zero out NaNs

        if unbiased:
            count = count - 1
            count = count.clamp(min=1)

        var = sq_diff.sum(dim=dim, keepdim=keepdim) / count
        return var.sqrt()

    @staticmethod
    def _fillna_with_ref(x: torch.Tensor, ref: torch.Tensor):
        for _ in range(x.ndim - ref.ndim):
            ref = ref.unsqueeze(0)  # now broadcastable
        return torch.where(torch.isnan(x), ref, x)

    def _norm_input_transform(self, x: torch.Tensor):
        eps = 1e-6
        norm_dyn_indexes = self.norm_dyn_indexes
        if len(norm_dyn_indexes) == 0:
            return x
        else:
            normed_x = deepcopy(x)
            normed_x[:, :, norm_dyn_indexes] = torch.log(
                normed_x[:, :, norm_dyn_indexes] + eps,
            )
            return normed_x

    def _norm_target_transform(self, x: torch.Tensor):
        eps = 1e-6
        if self.use_norm_target:
            return torch.log(x + eps)
        else:
            return x

    def _norm_input_inverse_transform(self, x: torch.Tensor):
        norm_dyn_indexes = self.norm_dyn_indexes
        if len(norm_dyn_indexes) == 0:
            return x
        else:
            denormed_x = deepcopy(x)
            denormed_x[:, :, norm_dyn_indexes] = torch.exp(
                denormed_x[:, :, norm_dyn_indexes],
            )
            return denormed_x

    def _norm_target_inverse_transform(self, x: torch.Tensor):
        if self.use_norm_target:
            return torch.exp(x)
        else:
            return x

    def fit(self, data: DistributedDataSchema) -> None:
        """Compute and store normalization statistics from a data chunk.

        Parameters
        ----------
        data
            A loaded data chunk whose mean and std will be stored for later
            use by ``transform`` and ``inverse_transform``.
        """
        dyn_input = self._norm_input_transform(data.dyn_input)
        self.mean['dyn_input'] = dyn_input.nanmean(dim=(0, 1))
        self.std['dyn_input'] = self._nanstd(dyn_input, dim=(0, 1))

        target = self._norm_target_transform(data.target)
        self.mean['target'] = target.nanmean()
        self.std['target'] = self._nanstd(target, dim=(0, 1))

        self.mean['static_input'] = data.static_input.nanmean(dim=0)
        self.std['static_input'] = self._nanstd(data.static_input, dim=0)

        self.mean['rout_static_input'] = data.rout_static_input.nanmean(dim=(0, 1))
        self.std['rout_static_input'] = self._nanstd(data.rout_static_input, dim=(0, 1))

    def transform(self, data: DistributedDataSchema) -> DistributedDataSchema:
        """Normalize a data chunk using stored mean/std statistics.

        Applies optional log-transforms and z-score normalization to dynamic
        inputs, static inputs, routing attributes, and targets. Returns a new
        schema with both raw and scaled fields populated.

        Parameters
        ----------
        data
            Raw data chunk to normalize.

        Returns
        -------
        DistributedDataSchema
            Copy of ``data`` with ``scaled_*`` fields populated.
        """
        eps = 1e-6
        # dynamic input
        dyn_input = self._norm_input_transform(data.dyn_input)
        dyn_input = (dyn_input - self.mean['dyn_input'].expand_as(dyn_input)) / (
            self.std['dyn_input'].expand_as(dyn_input) + eps
        )
        # target
        target = self._norm_target_transform(data.target)
        target = (target - self.mean['target'].expand_as(target)) / (
            self.std['target'].expand_as(target) + eps
        )
        # static input
        static_input = (
            data.static_input - self.mean['static_input'].expand_as(data.static_input)
        ) / (self.std['static_input'].expand_as(data.static_input) + eps)
        # rout static input
        rout_static_input = (
            data.rout_static_input
            - self.mean['rout_static_input'].expand_as(data.rout_static_input)
        ) / (self.std['rout_static_input'].expand_as(data.rout_static_input) + eps)
        return DistributedDataSchema(
            dyn_input=data.dyn_input,
            static_input=data.static_input,
            target=data.target,
            rout_static_input=data.rout_static_input,
            ac_all=data.ac_all,
            elev_all=data.elev_all,
            areas=data.areas,
            gauge=data.gauge,
            gauge_index=data.gauge_index,
            time=data.time,
            topo=data.topo,
            unit=data.unit,
            scaled_dyn_input=dyn_input,
            scaled_static_input=static_input,
            scaled_target=target,
            scaled_rout_static_input=rout_static_input,
        )

    def inverse_transform(
        self,
        tensor_data: torch.Tensor,
        varname: str,
    ) -> torch.Tensor:
        """Reverse normalization for a single variable.

        Applies the inverse z-score and, for ``'dyn_input'`` or ``'target'``,
        the inverse log-transform.

        Parameters
        ----------
        tensor_data
            Normalized tensor to denormalize.
        varname
            Key into the stored statistics (e.g. ``'dyn_input'``, ``'target'``).

        Returns
        -------
        torch.Tensor
            Denormalized tensor in original units.
        """
        descaled_data = tensor_data * self.std[varname].expand_as(
            tensor_data,
        ) + self.mean[varname].expand_as(tensor_data)
        if varname == 'dyn_input':
            descaled_data = self._norm_input_inverse_transform(descaled_data)
        elif varname == 'target':
            descaled_data = self._norm_target_inverse_transform(descaled_data)
        return descaled_data

    def fillna(self, data: DistributedDataSchema) -> DistributedDataSchema:
        """Replace NaN values in a data chunk with stored mean statistics.

        Parameters
        ----------
        data
            Data chunk that may contain NaN values in dynamic inputs, static
            inputs, or routing attributes.

        Returns
        -------
        DistributedDataSchema
            Copy of ``data`` with NaN values replaced by channel-wise means.
        """
        dyn_input = self._fillna_with_ref(data.dyn_input, self.mean['dyn_input'])
        static_input = self._fillna_with_ref(
            data.static_input,
            self.mean['static_input'],
        )
        rout_static_input = self._fillna_with_ref(
            data.rout_static_input,
            self.mean['rout_static_input'],
        )
        return DistributedDataSchema(
            dyn_input=dyn_input,
            static_input=static_input,
            target=data.target,
            rout_static_input=rout_static_input,
            ac_all=data.ac_all,
            elev_all=data.elev_all,
            areas=data.areas,
            gauge=data.gauge,
            gauge_index=data.gauge_index,
            time=data.time,
            topo=data.topo,
            unit=data.unit,
        )

    def save_stat(self, path: Union[str, Path]) -> None:
        """Serialize normalization statistics to a JSON file.

        Parameters
        ----------
        path
            Destination file path for the serialized statistics.
        """
        save_data = {
            'mean': {key: value.tolist() for key, value in self.mean.items()},
            'std': {key: value.tolist() for key, value in self.std.items()},
            'norm_dyn_indexes': self.norm_dyn_indexes,
            'use_norm_target': self.use_norm_target,
        }
        with open(path, 'w') as f:
            json.dump(save_data, f)

    def load_stat(self, path: Union[str, Path]) -> None:
        """Load normalization statistics from a JSON file.

        Parameters
        ----------
        path
            Path to the JSON file previously written by ``save_stat``.
        """
        with open(path) as f:
            load_data = json.load(f)
        self.mean = {
            key: torch.tensor(value) for key, value in load_data['mean'].items()
        }
        self.std = {key: torch.tensor(value) for key, value in load_data['std'].items()}
        self.norm_dyn_indexes = load_data['norm_dyn_indexes']
        self.use_norm_target = load_data['use_norm_target']

    def load_to_device(self, device: torch.device) -> None:
        """Move all stored statistic tensors to ``device``.

        Parameters
        ----------
        device
            Target device (e.g. ``torch.device('cuda')``).
        """
        for key in self.mean:
            self.mean[key] = self.mean[key].to(device)
            self.std[key] = self.std[key].to(device)

    def combine_chunk_stats(self, stats: list[dict]) -> None:
        """Merge per-chunk statistics into a single global mean and std.

        Uses the pooled variance formula to combine statistics computed
        independently over multiple data chunks.

        Parameters
        ----------
        stats
            List of per-chunk statistic dicts, each with keys ``'mean'``
            (dict of tensors), ``'std'`` (dict of tensors), and ``'count'``
            (int number of samples in the chunk).
        """
        total_count = sum([stat['count'] for stat in stats])
        combined_mean = {}
        combined_var = {}
        combined_std = {}

        for key in stats[0]['mean'].keys():
            # Combine means
            combined_mean[key] = (
                sum([stat['mean'][key] * stat['count'] for stat in stats]) / total_count
            )
            # Combine variances using the formula for pooled variance
            combined_var[key] = (
                sum(
                    [
                        (
                            (
                                stat['std'][key] ** 2
                                + (stat['mean'][key] - combined_mean[key]) ** 2
                            )
                            * stat['count']
                        )
                        for stat in stats
                    ],
                )
                / total_count
            )
            combined_std[key] = torch.sqrt(combined_var[key])
        self.mean = combined_mean
        self.std = combined_std
