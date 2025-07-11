import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch

log = logging.getLogger(__name__)

class FireSpatialTesting:
    """
    Spatial validation for fire occurrence prediction.
    
    Supports both random spatial holdouts and geographic regional holdouts
    based on latitude/longitude coordinates.
    """
    
    def __init__(self, config: Dict[str, Any], random_seed: int = 42):
        """
        Initialize fire spatial testing.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        random_seed : int
            Random seed for reproducible results
        """
        self.config = config
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Get spatial testing configuration
        self.test_config = config.get('test', {})
        self.spatial_type = self.test_config.get('spatial_type', 'random')  # 'random' or 'regional'
        self.n_folds = self.test_config.get('n_spatial_folds', 5)
        self.holdout_fraction = self.test_config.get('holdout_fraction', 0.2)
        
        log.info(f"Initialized FireSpatialTesting with {self.spatial_type} holdouts, {self.n_folds} folds")
    
    def create_spatial_folds(
        self, 
        dataset: Dict[str, torch.Tensor],
        lat_key: str = 'latitude',
        lon_key: str = 'longitude'
    ) -> List[Dict[str, Any]]:
        """
        Create spatial cross-validation folds for fire data.
        
        Parameters
        ----------
        dataset : Dict[str, torch.Tensor]
            Dataset containing grid cell data
        lat_key : str
            Key for latitude data in dataset
        lon_key : str
            Key for longitude data in dataset
            
        Returns
        -------
        List[Dict[str, Any]]
            List of fold configurations with train/test indices
        """
        # Extract coordinates from dataset
        lat, lon = self._extract_coordinates(dataset, lat_key, lon_key)
        n_cells = len(lat)
        
        log.info(f"Creating {self.n_folds} spatial folds for {n_cells} grid cells")
        log.info(f"Coordinate ranges: lat [{lat.min():.2f}, {lat.max():.2f}], lon [{lon.min():.2f}, {lon.max():.2f}]")
        
        if self.spatial_type == 'random':
            folds = self._create_random_spatial_folds(n_cells)
        elif self.spatial_type == 'regional':
            folds = self._create_regional_spatial_folds(lat, lon)
        else:
            raise ValueError(f"Unknown spatial_type: {self.spatial_type}")
        
        # Add fold metadata
        for i, fold in enumerate(folds):
            fold['fold_id'] = i
            fold['spatial_type'] = self.spatial_type
            fold['random_seed'] = self.random_seed
            fold['train_cells'] = len(fold['train_indices'])
            fold['test_cells'] = len(fold['test_indices'])
            
            log.info(f"Fold {i}: {fold['train_cells']} train cells, {fold['test_cells']} test cells")
        
        return folds
    
    def _extract_coordinates(
        self, 
        dataset: Dict[str, torch.Tensor],
        lat_key: str,
        lon_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract latitude and longitude coordinates from dataset."""
        
        # Try different ways to get coordinates
        lat, lon = None, None
        
        # Method 1: Direct keys in dataset
        if lat_key in dataset:
            lat = dataset[lat_key].cpu().numpy()
            if lat.ndim > 1:
                lat = lat.flatten()
        if lon_key in dataset:
            lon = dataset[lon_key].cpu().numpy()
            if lon.ndim > 1:
                lon = lon.flatten()
        
        # Method 2: Try common alternative names
        if lat is None:
            for key in ['lat', 'latitude', 'y']:
                if key in dataset:
                    lat = dataset[key].cpu().numpy()
                    if lat.ndim > 1:
                        lat = lat.flatten()
                    break
        
        if lon is None:
            for key in ['lon', 'longitude', 'x']:
                if key in dataset:
                    lon = dataset[key].cpu().numpy()
                    if lon.ndim > 1:
                        lon = lon.flatten()
                    break
        
        # Method 3: Extract from static attributes if available
        if lat is None or lon is None:
            if 'c_nn' in dataset:
                static_data = dataset['c_nn'].cpu().numpy()
                # Assuming lat/lon are first two columns based on your config
                if static_data.shape[1] >= 2:
                    if lat is None:
                        lat = static_data[:, 0]  # Assuming first column is lat
                    if lon is None:
                        lon = static_data[:, 1]  # Assuming second column is lon
        
        if lat is None or lon is None:
            raise ValueError("Could not extract latitude/longitude coordinates from dataset")
        
        if len(lat) != len(lon):
            raise ValueError(f"Coordinate mismatch: lat has {len(lat)} points, lon has {len(lon)} points")
        
        return lat, lon
    
    def _create_random_spatial_folds(self, n_cells: int) -> List[Dict[str, Any]]:
        """Create random spatial holdout folds."""
        log.info("Creating random spatial folds")
        
        # Create indices for all grid cells
        all_indices = np.arange(n_cells)
        
        # Shuffle indices with fixed seed for reproducibility
        np.random.seed(self.random_seed)
        shuffled_indices = np.random.permutation(all_indices)
        
        # Calculate fold sizes
        fold_size = n_cells // self.n_folds
        
        folds = []
        for fold_idx in range(self.n_folds):
            # Calculate test indices for this fold
            start_idx = fold_idx * fold_size
            if fold_idx == self.n_folds - 1:
                # Last fold gets remaining indices
                end_idx = n_cells
            else:
                end_idx = (fold_idx + 1) * fold_size
            
            test_indices = shuffled_indices[start_idx:end_idx]
            train_indices = np.concatenate([
                shuffled_indices[:start_idx],
                shuffled_indices[end_idx:]
            ])
            
            folds.append({
                'train_indices': train_indices,
                'test_indices': test_indices,
                'description': f'Random spatial fold {fold_idx}'
            })
        
        return folds
    
    def _create_regional_spatial_folds(self, lat: np.ndarray, lon: np.ndarray) -> List[Dict[str, Any]]:
        """Create geographic regional holdout folds based on lat/lon clustering."""
        log.info("Creating regional spatial folds using geographic clustering")
        
        # Combine coordinates for clustering
        coords = np.column_stack([lat, lon])
        
        # Use KMeans to create geographic regions
        np.random.seed(self.random_seed)
        kmeans = KMeans(n_clusters=self.n_folds, random_state=self.random_seed, n_init=10)
        region_labels = kmeans.fit_predict(coords)
        
        # Log region statistics
        for region_id in range(self.n_folds):
            region_mask = region_labels == region_id
            region_count = np.sum(region_mask)
            region_lat = lat[region_mask]
            region_lon = lon[region_mask]
            
            log.info(f"Region {region_id}: {region_count} cells, "
                    f"lat [{region_lat.min():.2f}, {region_lat.max():.2f}], "
                    f"lon [{region_lon.min():.2f}, {region_lon.max():.2f}]")
        
        # Create folds where each region is held out once
        folds = []
        for fold_idx in range(self.n_folds):
            test_mask = region_labels == fold_idx
            train_mask = ~test_mask
            
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(train_mask)[0]
            
            # Get region center for description
            test_lat = lat[test_indices]
            test_lon = lon[test_indices]
            center_lat = np.mean(test_lat)
            center_lon = np.mean(test_lon)
            
            folds.append({
                'train_indices': train_indices,
                'test_indices': test_indices,
                'region_id': fold_idx,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'description': f'Regional fold {fold_idx} (center: {center_lat:.2f}, {center_lon:.2f})'
            })
        
        return folds
    
    def split_dataset_by_fold(
        self, 
        dataset: Dict[str, torch.Tensor], 
        fold: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Split dataset into train/test based on fold configuration.
        
        Parameters
        ----------
        dataset : Dict[str, torch.Tensor]
            Full dataset
        fold : Dict[str, Any]
            Fold configuration with train/test indices
            
        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Train dataset, test dataset
        """
        train_indices = fold['train_indices']
        test_indices = fold['test_indices']
        
        train_dataset = {}
        test_dataset = {}
        
        for key, tensor in dataset.items():
            if tensor.dim() >= 2 and tensor.shape[1] == len(train_indices) + len(test_indices):
                # This tensor has a spatial dimension that matches total cells
                if tensor.dim() == 2:  # [time, cells] or [cells, features]
                    if tensor.shape[0] > tensor.shape[1]:  # Likely [time, cells]
                        train_dataset[key] = tensor[:, train_indices]
                        test_dataset[key] = tensor[:, test_indices]
                    else:  # Likely [cells, features]
                        train_dataset[key] = tensor[train_indices, :]
                        test_dataset[key] = tensor[test_indices, :]
                elif tensor.dim() == 3:  # [time, cells, features]
                    train_dataset[key] = tensor[:, train_indices, :]
                    test_dataset[key] = tensor[:, test_indices, :]
                else:
                    # For higher dimensions, assume second dimension is spatial
                    train_dataset[key] = tensor.index_select(1, torch.tensor(train_indices, device=tensor.device))
                    test_dataset[key] = tensor.index_select(1, torch.tensor(test_indices, device=tensor.device))
            else:
                # This tensor doesn't have spatial dimension or doesn't match expected size
                # Keep the same for both train and test (e.g., temporal features)
                train_dataset[key] = tensor
                test_dataset[key] = tensor
        
        log.info(f"Split dataset: {len(train_indices)} train cells, {len(test_indices)} test cells")
        
        return train_dataset, test_dataset


def run_fire_spatial_testing(config: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Run spatial cross-validation for fire occurrence prediction.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    model : torch.nn.Module
        Fire prediction model
        
    Returns
    -------
    Dict[str, Any]
        Results from all spatial folds
    """
    from dmg.core.utils.factory import import_data_loader, import_trainer
    
    log.info("Starting fire spatial cross-validation")
    
    # Initialize spatial testing
    spatial_tester = FireSpatialTesting(config)
    
    # Load full dataset (no splitting yet)
    data_loader_cls = import_data_loader(config['data_loader'])
    data_loader = data_loader_cls(config, test_split=False, overwrite=False)
    full_dataset = data_loader.dataset
    
    # Create spatial folds
    folds = spatial_tester.create_spatial_folds(full_dataset)
    
    # Results storage
    all_results = {
        'config': config,
        'spatial_type': spatial_tester.spatial_type,
        'n_folds': len(folds),
        'random_seed': spatial_tester.random_seed,
        'fold_results': []
    }
    
    # Run each fold
    for fold_idx, fold in enumerate(folds):
        log.info(f"Running spatial fold {fold_idx + 1}/{len(folds)}: {fold['description']}")
        
        try:
            # Split dataset for this fold
            train_dataset, test_dataset = spatial_tester.split_dataset_by_fold(full_dataset, fold)
            
            # Create trainer for this fold
            trainer_cls = import_trainer(config['trainer'])
            trainer = trainer_cls(
                config,
                model,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                verbose=False,  # Reduce verbosity for cross-validation
            )
            
            # Train model on this fold
            if 'train' in config['mode']:
                trainer.train()
            
            # Evaluate on held-out spatial region
            if 'test' in config['mode']:
                batch_predictions, observations = trainer.evaluate()
                
                # Store fold results
                fold_result = {
                    'fold_id': fold_idx,
                    'fold_info': fold,
                    'train_cells': fold['train_cells'],
                    'test_cells': fold['test_cells'],
                    'metrics_path': trainer.config.get('out_path', 'results'),
                }
                
                all_results['fold_results'].append(fold_result)
                
                log.info(f"Completed fold {fold_idx + 1}: "
                        f"{fold['train_cells']} train cells, {fold['test_cells']} test cells")
            
        except Exception as e:
            log.error(f"Error in spatial fold {fold_idx}: {str(e)}")
            continue
    
    log.info(f"Fire spatial cross-validation completed: {len(all_results['fold_results'])} successful folds")
    
    return all_results