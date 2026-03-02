# Contributing to Machine Boss

## Building and testing

### C++

```bash
brew install gsl boost htslib pkg-config   # macOS
make
npm install
make test
```

Requires clang++ or g++ with C++11 support. Tests use `t/testexpect.py` and require Node.js.

### Python

```bash
cd python
pip install -e '.[jax]'
pip install pytest
pytest machineboss/
```

Requires Python 3.10+. The `[jax]` extra is needed for JAX-based tests.

## Pull request workflow

1. Fork the repository
2. Create a feature branch from `master`
3. Make your changes
4. Run both C++ and Python tests
5. Open a pull request against `master`

## Code style

- **C++**: C++11, consistent with existing `src/` conventions
- **Python**: Python 3.10+, tested with pytest
- Keep commits focused and well-described

## Reporting bugs

Open an issue at https://github.com/evoldoers/machineboss/issues with:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, compiler, Python version)
