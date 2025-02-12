## Differentiable Parameter Learning in Hydrology

This directory contains notebooks to train/forward 3 published differentiable hydrologic models in order of development:

1. $\delta$ HBV 1.0 -- `example_dhbv_1_0.ipynb`
2. $\delta$ HBV 1.1p -- `example_dhbv_1_1p.ipynb`
3. $\delta$ HBV 2.0UH -- `example_dhbv_2_0.ipynb` (Forward only)

<br>

We encourage looking at $\delta$ HBV 1.0 first for detailed on differentiable modeling and dMG methods.

All of these models rely on the bucket-based hydrology model HBV (Beck 2020; Seibert 2005) for making streamflow predictions.

1 and 2 contain code to train, test, and forward benchmark versions of these models. $\delta$ HBV 2.0UH training and testing codes will be made available at a later date.
