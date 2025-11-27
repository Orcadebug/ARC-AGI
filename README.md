# Emergent-ARC v2.0

**Emergent-ARC** is a neurosymbolic architecture for solving abstract reasoning tasks (ARC-AGI). It evolves compact Spiking Neural Networks (SNNs) that compose programs from a physics-inspired Domain Specific Language (DSL).

## Features

- **Object-Centric Perception**: Reduces grid to object slots.
- **Hierarchical DSL**: Physics, relational, and generative primitives.
- **Compact SNN Policy**: ~12k parameters, efficient evolution.
- **Online Program Induction**: Test-time hypothesis validation.
- **Veteran Population**: Continuous knowledge transfer.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Structure

- `emergent_arc/`: Core package
  - `detection/`: Object detection and feature extraction
  - `dsl/`: Domain Specific Language primitives and executor
  - `network/`: Spiking Neural Network architecture
  - `evolution/`: Evolutionary algorithms (PGPE)
  - `memory/`: Subroutine library and veteran pool
  - `inference/`: Online program induction
- `tests/`: Unit tests
- `notebooks/`: Analysis and exploration
