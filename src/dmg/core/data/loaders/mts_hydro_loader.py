import json
import logging
from typing import Any, Union, list
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from pathlib import Path
import xarray as xr
import geopandas as gpd
import networkx as nx
from copy import deepcopy

from dmg.core.data.loaders.base import BaseLoader

from distributedDS.core.data.utils.topo_operator import (
    reachability_matrix,
    PathWeightedAgg,
)

log = logging.getLogger(__name__)


class MtsHydroLoader(BaseLoader):
    """MTS hydrological data loader."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config

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
        """
        Load dataset for the specified mode ('train', 'valid', 'test') to self.dataset. 'test' is also used for simulation
        If number of chunks is 1, load the entire dataset into memory. Otherwise, use a generator to yield data chunk by chunk.
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
                "mode should be one of ['train', 'valid', 'test', 'simulation']"
            )

    def get_dataset(self):
        """Yield preprocessed data from the loaded dataset."""
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Please call load_dataset() first.")
        for data in self.dataset:
            yield self.preprocessor.transform(data)

    def _preprocess_data(self) -> dict[str, torch.Tensor]:
        """Read, preprocess, and return data as dictionary of torch tensors."""
        print("Preprocessing data...")

    def load_norm_stats(self) -> None:
        """Load normalization statistics from preprocessing paths."""
        try:
            preprocessing_paths = self.config['observations']['preprocessing']
            with open(Path(preprocessing_paths['path_stats']), 'rb') as f:
                self.stats_dict = json.load(f)
            self.preprocessor = DistributedDataPreprocessor()
            self.preprocessor.load_stat(Path(preprocessing_paths['path_preprocess']))
        except ValueError as e:
            raise ValueError("Error loading normalization statistics.") from e

    def cleanup_memory(self) -> None:
        """Clean up loaded datasets to free memory."""
        self.dataset = None


class DistributedDataSchema(BaseModel):
    """MTS data schema."""

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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DistributedDataReader:
    """MTS data reader."""

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
    ):
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
            start=runoff_start_time, periods=xr_runoff['time'].shape[0], freq='h'
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
                gauge_info.loc[gauge_info['DRAIN_SQKM'] < area_thres, 'gauge_id']
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
            }
        )
        missing_gauges = df_map.loc[
            df_map['unit'].isin(set(df_map['unit']) - set(df_topo['unit'])), 'gauge'
        ]
        df_topo = df_topo[~df_topo['gauge'].isin(missing_gauges)].reset_index(drop=True)
        df_topo = df_topo.sort_values(by=['gauge', 'unit']).reset_index(drop=True)
        df_topo['is_upstream'] = 1
        df_topo = df_topo.pivot_table(
            index='gauge', columns='unit', values='is_upstream', fill_value=0
        )
        selected_gauges = df_topo.index.values
        selected_basins = df_topo.columns.values

        return selected_gauges, selected_basins

    def get_element_ids(self, area_thres: float, years: list[int]):
        """Get selected gauge and basin ids based on area threshold and years."""
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
        def get_element_indexes(
            element_array: np.ndarray, elements: Union[np.ndarray, list]
        ) -> np.ndarray:
            df = pd.DataFrame(
                {'element': element_array, 'local_ind': np.arange(len(element_array))}
            )
            df = df.merge(
                pd.DataFrame(
                    {'element': elements, 'global_ind': np.arange(len(elements))}
                )
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
            [xr_attrs['attr'].data.tolist().index(key) for key in attribute_order]
        )
        attr_names = xr_attrs['attr'].data[attrs_indexes]
        attrs = xr_attrs['__xarray_dataarray_variable__'].data[attrs_indexes, :][
            :, attrs_duplicated_indexes
        ]
        xr_attrs.close()

        # read runoff data
        xr_runoff = xr.open_dataset(path_runoff)
        runoff_times = pd.date_range(
            start=runoff_start_time, periods=xr_runoff['time'].shape[0], freq='h'
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
                start=f'{years[0]}-01-01', periods=runoff.shape[1], freq='h'
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
            min=0.0
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
            divides['divide_id'], 'cat-' + pd.Series(selected_basins).astype(str)
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
        """Read distributed hourly data for the specified years."""
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
        """Predict targets within start_year and end_year (inclusive), with warmup days before each chunk."""
        chunk_starts = list(range(start_year, end_year + 1, self.chunk_year_size))
        if shuffle:
            perm = torch.randperm(len(chunk_starts)).tolist()
            chunk_starts = [chunk_starts[i] for i in perm]
        for i in chunk_starts:
            years = list(range(i, min(i + self.chunk_year_size, end_year + 1)))
            warmup_dates = pd.date_range(
                end=f'{years[0]}-01-01', periods=self.warmup_days + 1, freq='d'
            )[:-1]
            warm_years = warmup_dates.year.unique().tolist()
            years = warm_years + years
            data = self.read_distributed_hourly_data(years)
            pre_read_dates = pd.date_range(
                start=f'{warm_years[0]}-01-01', end=f'{warm_years[-1]}-12-31', freq='d'
            )
            start_time_index = (len(pre_read_dates) - len(warmup_dates)) * 24
            data.target = data.target[:, start_time_index:]
            data.dyn_input = data.dyn_input[:, start_time_index:, :]
            data.time = data.time[start_time_index:]
            yield data

    def yield_train_set(self):
        """Get training set."""
        if self.train_start_year is None or self.train_end_year is None:
            raise ValueError("train years are not specified.")
        yield from self.yield_chunk_set(
            self.train_start_year, self.train_end_year, shuffle=True
        )

    def yield_valid_set(self):
        """Get validation set."""
        if self.valid_start_year is None or self.valid_end_year is None:
            raise ValueError("validation years are not specified.")
        yield from self.yield_chunk_set(
            self.valid_start_year, self.valid_end_year, shuffle=False
        )

    def yield_test_set(self):
        """Get test set."""
        if self.test_start_year is None or self.test_end_year is None:
            raise ValueError("test years are not specified.")
        yield from self.yield_chunk_set(
            self.test_start_year, self.test_end_year, shuffle=False
        )


class DistributedDataPreprocessor:
    """MTS preprocessor."""

    def __init__(
        self, norm_dyn_indexes: list[int] = None, use_norm_target: bool = False
    ):
        self.mean = {}
        self.std = {}
        self.norm_dyn_indexes = norm_dyn_indexes
        self.use_norm_target = use_norm_target

    @staticmethod
    def _nanstd(x: torch.Tensor, dim: Union[int, list], keepdim=False, unbiased=True):
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
                normed_x[:, :, norm_dyn_indexes] + eps
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
                denormed_x[:, :, norm_dyn_indexes]
            )
            return denormed_x

    def _norm_target_inverse_transform(self, x: torch.Tensor):
        if self.use_norm_target:
            return torch.exp(x)
        else:
            return x

    def fit(self, data: DistributedDataSchema):
        """Get stats."""
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
        """MTS transform."""
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
        self, tensor_data: torch.Tensor, varname: str
    ) -> torch.Tensor:
        """Transform."""
        descaled_data = tensor_data * self.std[varname].expand_as(
            tensor_data
        ) + self.mean[varname].expand_as(tensor_data)
        if varname == 'dyn_input':
            descaled_data = self._norm_input_inverse_transform(descaled_data)
        elif varname == 'target':
            descaled_data = self._norm_target_inverse_transform(descaled_data)
        return descaled_data

    def fillna(self, data: DistributedDataSchema):
        """Fill nans."""
        dyn_input = self._fillna_with_ref(data.dyn_input, self.mean['dyn_input'])
        static_input = self._fillna_with_ref(
            data.static_input, self.mean['static_input']
        )
        rout_static_input = self._fillna_with_ref(
            data.rout_static_input, self.mean['rout_static_input']
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

    def save_stat(self, path: Union[str, Path]):
        """Save stats."""
        save_data = {
            'mean': {key: value.tolist() for key, value in self.mean.items()},
            'std': {key: value.tolist() for key, value in self.std.items()},
            'norm_dyn_indexes': self.norm_dyn_indexes,
            'use_norm_target': self.use_norm_target,
        }
        json.dump(save_data, open(path, 'w'))

    def load_stat(self, path: Union[str, Path]):
        """Load stats."""
        load_data = json.load(open(path))
        self.mean = {
            key: torch.tensor(value) for key, value in load_data['mean'].items()
        }
        self.std = {key: torch.tensor(value) for key, value in load_data['std'].items()}
        self.norm_dyn_indexes = load_data['norm_dyn_indexes']
        self.use_norm_target = load_data['use_norm_target']

    def load_to_device(self, device: torch.device):
        """Load to device."""
        for key in self.mean:
            self.mean[key] = self.mean[key].to(device)
            self.std[key] = self.std[key].to(device)

    def combine_chunk_stats(self, stats: list[dict]):
        """
        :param stats: [{'mean': dict, 'std': dict, 'count': int}].
        :return:
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
                    ]
                )
                / total_count
            )
            combined_std[key] = torch.sqrt(combined_var[key])
        self.mean = combined_mean
        self.std = combined_std
