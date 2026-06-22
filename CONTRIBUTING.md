# Contributing to PULSE

Thank you for your interest in contributing to the PULSE Simulator! This project is developed at [IRISA Laboratory](https://www.irisa.fr) as part of a PhD research project on UWB indoor localization.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/moussaart/pulse-simulator/issues) to report bugs or suggest features.
- Include your OS, Python version, and steps to reproduce the issue.

### Code Contributions

1. **Fork** the repository.
2. Create a **feature branch**: `git checkout -b feature/my-feature`
3. Make your changes following the guidelines below.
4. **Test** your changes locally.
5. Submit a **Pull Request** with a clear description.

### Code Style

- Follow **PEP 8** for Python code.
- Use **type hints** for all function signatures.
- Add **docstrings** to all new classes and public methods.
- Keep imports organized: standard library → third-party → local.

### Adding Custom Algorithms

The simplest way to contribute is to add a new localization algorithm:

1. Create `src/user_algorithms/my_algo.py`
2. Inherit from `BaseLocalizationAlgorithm`
3. Implement `name`, `initialize()`, and `update()` methods

See the [Developer Guide](https://moussaart.github.io/pulse-simulator/documentation/advanced.html) for details.

## Development Setup

```bash
git clone https://github.com/moussaart/pulse-simulator.git
cd pulse-simulator
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py
```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
