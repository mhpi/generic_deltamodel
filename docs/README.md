# 𝛿MG Documentation

Welcome to the documentation for **𝛿MG** (Differentiable Model Generator), a PyTorch framework for building differentiable models that couple neural networks with process-based equations.

𝛿MG is designed to scale with modern deep learning tools (e.g., foundation models) while maintaining physical interpretability. Our peer-reviewed and published [benchmarks](https://mhpi.github.io/benchmarks/#10-year-training-comparison) show that well-tuned differentiable models can match deep networks in performance — while better extrapolating to extreme or data-scarce conditions and predicting physically meaningful variables.

Differentiable modeling introduces more modeling choices than traditional deep learning due to its physical constraints. This includes learning parameters, missing process representations, corrections, or other enhancements for physical models.

> **Note**: Differentiable models come with a larger decision space than purely data-driven neural networks since physical processes are involved, and can thus feel "trickier" to work with. We recommend beginning with the example [notebooks](../example/hydrology/) and systematically making changes, one at a time. Pay attention to multifaceted outputs, diverse causal analyses, and predictions of untrained variables permitted by differentiable models, rather than purely trying to outperform other models' metrics.

## Getting Started

| Document | Description |
|----------|-------------|
| [Setup](./setup.md) | Installation from PyPI or source, environment setup, optional dependencies |
| [How to Run](./how_to_run.md) | Running experiments from the command line and building custom models |
| [Configuration](./configuration.md) | Configuration file system, all available settings, and glossary |

## API

| Document | Description |
|----------|-------------|
| [API Reference](./api_reference.md) | Public API catalog — models, loss functions, neural networks, utilities |

## Contributing

| Document | Description |
|----------|-------------|
| [Contributing Guide](./CONTRIBUTING.md) | How to fork, test, lint, and submit pull requests |
| [Style Guide](./style_guide.md) | Code conventions, docstring format, type hint standards |
| [Attributions](./attributions.md) | Code attribution and contributor credits |

## Additional Resources

| Resource | Description |
|----------|-------------|
| [Examples](../example/hydrology/) | Jupyter notebook tutorials for hydrology use cases |
| [Changelog](./CHANGELOG.md) | Release history and version notes |
| [GitHub Issues](https://github.com/mhpi/generic_deltamodel/issues) | Bug reports, questions, and feature requests |
