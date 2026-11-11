import numpy as np
import torch
from torch import nn


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(
        self,
        mask_ratio_time_series,
        mask_ratio_static=None,
        min_window_size=None,
        max_window_size=None,
    ):
        super().__init__()
        self.mask_ratio_time_series = mask_ratio_time_series
        self.mask_ratio_static = mask_ratio_static
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

    def isolated_point_masking(self, data_shape):
        batch_size, num_features = data_shape
        batch_masked_index_list = []
        for _ in np.arange(batch_size):
            masked_index = self.one_batch_isolated_point_masking(
                num_features, self.mask_ratio_static
            )
            batch_masked_index_list.append(masked_index)

        batch_masked_index_list = np.stack(batch_masked_index_list, axis=0)
        batch_unmasked_index_list = np.logical_not(batch_masked_index_list)

        # convert to torch
        masked_index = torch.from_numpy(batch_masked_index_list)
        unmasked_index = torch.from_numpy(batch_unmasked_index_list)

        return unmasked_index, masked_index

    @staticmethod
    def one_batch_group_isolated_point_masking(
        mask_ratio, features_name, group_mask_dict
    ):
        # Number of features in the dataset
        num_features = len(features_name)
        # Initializing a boolean array for masking, all set to False initially
        masked_index = np.zeros(num_features, dtype=bool)

        # Calculate the total number of 'effective' features, where each group is counted as one feature
        # and all non-group features are counted individually
        num_groups_as_single = len(group_mask_dict)
        non_group_features = len(features_name) - sum(
            len(group) for group in group_mask_dict.values()
        )
        total_effective_features = num_groups_as_single + non_group_features

        # Calculate the number of points/features to mask based on the new total and the mask ratio
        num_masked_points = int(total_effective_features * mask_ratio)
        assert num_masked_points > 0, (
            "Mask ratio is too small or number of features is too small!"
        )

        # Mapping feature names to their indices for easy access
        feature_to_index = {name: i for i, name in enumerate(features_name)}

        # Processing each group for masking
        for group in group_mask_dict.values():
            if num_masked_points <= 0:
                break

            # Randomly decide if this group is to be masked
            if np.random.rand() < mask_ratio:
                # If so, reduce the count of features to mask
                num_masked_points -= 1
                # Mask all features in the group
                for feature in group:
                    index = feature_to_index[feature]
                    masked_index[index] = True

        # If there are still features to mask after processing groups
        if num_masked_points > 0:
            # Create a list of features not in any group
            non_group_features_list = [
                name
                for name in features_name
                if all(name not in group for group in group_mask_dict.values())
            ]
            np.random.shuffle(non_group_features_list)
            # Mask features from this list until no more features need to be masked
            for feature in non_group_features_list:
                if num_masked_points <= 0:
                    break
                index = feature_to_index[feature]
                if not masked_index[index]:
                    masked_index[index] = True
                    num_masked_points -= 1

        return masked_index

    @staticmethod
    def one_batch_isolated_point_masking(num_features, mask_ratio):
        masked_index = np.zeros(num_features, dtype=bool)
        num_masked_points = int(num_features * mask_ratio)
        assert num_masked_points > 0, (
            "Mask ratio is too small or number of features is too small!"
        )
        masked_index[:num_masked_points] = True
        np.random.shuffle(masked_index)
        return masked_index

    def consecutive_masking(self, data_shape):
        batch_size, seq_length, num_features = data_shape

        batch_masked_index_list = []
        for _ in np.arange(batch_size):
            masked_index = self.one_batch_consecutive_with_variables(
                seq_length,
                num_features,
                self.mask_ratio_time_series,
                self.min_window_size,
                self.max_window_size,
            )
            batch_masked_index_list.append(masked_index)

        batch_masked_index = np.stack(batch_masked_index_list, axis=0)
        batch_unmaksed_index = np.logical_not(batch_masked_index)

        # convert to torch
        masked_index = torch.from_numpy(batch_masked_index)
        unmasked_index = torch.from_numpy(batch_unmaksed_index)

        return unmasked_index, masked_index

    @staticmethod
    def one_batch_consecutive_with_variables(
        seq_length, num_features, mask_ratio, min_window_size=None, max_window_size=None
    ):

        masked_index = np.zeros((seq_length, num_features), dtype=np.bool)

        for feature in np.arange(num_features):
            total_masked = 0
            while total_masked < int(seq_length * mask_ratio):
                remaining = seq_length - total_masked
                window_size = np.random.randint(
                    min_window_size, min(max_window_size, remaining) + 1
                )
                start = np.random.randint(0, seq_length - window_size + 1)
                end = start + window_size

                # Update the masked_index and the total masked count for this feature
                masked_index[start:end, feature] = True
                total_masked += window_size

                # Adjust for overshoot
                if total_masked > int(seq_length * mask_ratio):
                    overshoot = total_masked - int(seq_length * mask_ratio)
                    masked_index[end - overshoot : end, feature] = False
                    total_masked -= overshoot

        return masked_index

    def forward(self, data_shape, method="consecutive"):
        if method in ["consecutive"]:
            self.unmasked_index, self.masked_index = self.consecutive_masking(
                data_shape
            )
        elif method in ["isolated_point", "point"]:
            self.unmasked_index, self.masked_index = self.isolated_point_masking(
                data_shape
            )
        else:
            raise ValueError("Unknown masking method!")
        return self.unmasked_index, self.masked_index


def skip_or_mask_all_variables(
    masked_index, features_order, mask_skip_variables, mask_all_variables, mode
):
    if mask_skip_variables is not None:
        assert features_order is not None, (
            "features_order should not be None when mask_skip_variables is not None"
        )
        # get the index of the skip variables
        skip_index = np.array(
            [
                features_order.index(var)
                for var in mask_skip_variables
                if var in features_order
            ]
        )
        if len(skip_index) > 0:
            masked_index[..., skip_index] = False

    if mode in ["test"]:
        if mask_all_variables is not None:
            assert features_order is not None, (
                "features_order should not be None when mask_all_variables is not None"
            )
            # get the index of the skip variables
            all_index = np.array(
                [
                    features_order.index(var)
                    for var in mask_all_variables
                    if var in features_order
                ]
            )
            if len(all_index) > 0:
                masked_index[..., all_index] = True

    return masked_index


def get_batch_mask(num_batch, config):
    mask_time_series_index_list, mask_static_index_list = [], []
    for i in range(num_batch):
        # size [seq_len, num_features]
        mask_time_series_index = MaskGenerator.one_batch_consecutive_with_variables(
            config.seq_len,
            len(config.time_series_variables),
            config.mask_ratio_time_series,
            config.min_window_size,
            config.max_window_size,
        )
        # size: [num_features]
        if config.group_mask_dict is None:
            mask_static_index = MaskGenerator.one_batch_isolated_point_masking(
                len(config.static_variables), config.mask_ratio_static
            )
        else:
            mask_static_index = MaskGenerator.one_batch_group_isolated_point_masking(
                config.mask_ratio_static,
                config.static_variables,
                config.group_mask_dict,
            )

        mask_time_series_index = skip_or_mask_all_variables(
            mask_time_series_index,
            config.time_series_variables,
            config.mask_skip_variables,
            config.mask_all_variables,
            config.mode,
        )
        mask_static_index = skip_or_mask_all_variables(
            mask_static_index,
            config.static_variables,
            config.mask_skip_variables,
            config.mask_all_variables,
            config.mode,
        )

        mask_time_series_index_list.append(mask_time_series_index)
        mask_static_index_list.append(mask_static_index)

    mask_time_series_index_list = np.stack(mask_time_series_index_list, axis=0)
    mask_static_index_list = np.stack(mask_static_index_list, axis=0)
    return mask_time_series_index_list, mask_static_index_list
