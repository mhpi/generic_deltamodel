# ML & Differentiable Parameter Learning (dPL) in Hydrology

This directory contains notebooks to train/test/forward 3 published differentiable hydrologic models (@Dr. Chaopeng Shen, MHPI) in order of development:

1. HydroDL LSTM -- `example_lstm.ipynb`
2. δHBV 1.0 -- `example_dhbv.ipynb`
3. δHBV 1.1p -- `example_dhbv_1_1p.ipynb`
4. δHBV 1.1p w/ GEFS -- `example_dhbv_1_1p_gefs.ipynb`
5. δHBV 2.0UH -- `example_dhbv_2.ipynb` (forward simulation only)
6. δHBV 2.0UH MTS -- `_example_dhbv2_mts.ipynb` (forward simulation; note this script is WIP and currently for development only.)

We encourage you to check δHBV 1.0 first for detailed on differentiable modeling and dMG methods.

All of these models rely on the bucket-based hydrology model HBV (Beck 2020; Seibert 2005) for making streamflow predictions.

(1 - 4) contain code to train, test, and forward benchmark versions of these models. Distributed model δHBV 2.0UH training and testing codes will be made available at a later date.
