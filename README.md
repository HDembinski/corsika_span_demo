# corsika_span_demo

A demonstrator for the ParticleSpan proposal.

# Installation notes

This repository uses git submodules, therefore you need to clone it with the `--recursive` option.
```sh
git clone --recursive https://github.com/HDembinski/corsika_span_demo.git
```

If you already have a clone, you can download the submodules with this command
```sh
git submodule update --init
```

# Building instructions

```sh
cmake .
make -j4
```

# Run benchmark and produce json output

```sh
./span_demo --benchmark_repetitions=3 --benchmark_out=perf.dat
```

# Plot json output

You need the following extra packages to run the plotting script: pyyaml, numpy, matplotlib.

```sh
python3 plot.py
```