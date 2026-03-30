# Specific Test II: Agentic AI

This task is implemented in a separate repository:

<https://github.com/virajvekaria/DeepLenseSim>

This folder stays lightweight on purpose, but the implementation behind it is not generic. I inspected the preserved local backup and this task already contains a concrete `deeplense_agent` package that wraps DeepLenseSim with a natural-language simulation workflow.

## What Is Implemented

The codebase includes:

- `deeplense_agent/models.py`: typed Pydantic request, plan, capability, artifact, and run-result schemas
- `deeplense_agent/agent.py`: the Pydantic AI agent, system prompt, provider stack, and tool registration
- `deeplense_agent/simulator.py`: the execution service that resolves defaults, runs DeepLenseSim, and writes outputs
- `deeplense_agent/cli.py`: an interactive and one-shot command-line interface exposed as `deeplense-agent`
- `tests/test_deeplense_agent.py`: smoke tests for preview validation and artifact generation

## Agent Design

This is a Pydantic AI implementation and the workflow is human-in-the-loop:

- the agent interprets a free-form prompt into a typed `SimulationRequest`
- if a crucial field is missing, it asks a concise follow-up question
- it calls `preview_simulation_plan(...)` first to resolve defaults and summarize the exact run
- it waits for explicit user confirmation
- only then does it call `run_deeplense_simulation(...)`

The registered tools in the code are:

- `get_supported_configurations`
- `preview_simulation_plan`
- `run_deeplense_simulation`

## Supported Configurations

The implemented agent currently supports:

- `Model_I`
- `Model_II`
- `Model_III`

Supported substructure families:

- `no_sub`
- `cdm`
- `axion`

`Model_IV` is intentionally excluded in the agent layer because it depends on the external Galaxy10 DECals dataset, which is not bundled with the repo.

## Typed Simulation Parameters

The `SimulationRequest` schema in the backup code supports structured fields such as:

- `model_config`
- `substructure_type`
- `image_count`
- `lens_redshift`
- `source_redshift`
- `resolution`
- `main_halo_mass`
- `axion_mass`
- `vortex_mass`
- `cdm_subhalo_mean`
- `output_root`
- `run_name`
- `seed`

The code also validates that `source_redshift` must be greater than `lens_redshift`.

## Execution Details

The simulation service maps the validated request into DeepLenseSim presets:

- `Model_I`: `150 x 150`, Gaussian PSF, amplitude-based Sersic source
- `Model_II`: `64 x 64`, Euclid-like instrument, magnitude-based source
- `Model_III`: `64 x 64`, HST-like instrument, magnitude-based source

When axion structure is requested but masses are omitted, the service defaults to:

- `axion_mass = 1e-23`
- `vortex_mass = 3e10`

## Output Artifacts

Each completed run writes structured outputs to disk:

- one `.npy` file per generated image
- one `.png` preview per generated image
- one `run_metadata.json` file with the full run description
- one `contact_sheet.png` summarizing the run

The returned `SimulationRunResult` also records image statistics such as shape, min, max, mean, and standard deviation.

## Included Sample Run

A full sample run is now copied directly into this folder at:

- `20260324_174713_axion-euclid-run/`

This bundled run shows the exact artifact layout produced by the agent and includes:

- `20260324_174713_axion-euclid-run/run_metadata.json`
- `20260324_174713_axion-euclid-run/contact_sheet.png`
- `20260324_174713_axion-euclid-run/image_000.png`
- `20260324_174713_axion-euclid-run/image_000.npy`

The included example corresponds to:

- `1` generated image
- `Model_II`
- `axion` substructure
- `64 x 64` Euclid-style output
- deterministic output with a recorded seed and structured metadata

## Runtime And CLI

The package exposes a CLI entry point:

- `deeplense-agent`

The code supports both interactive use and one-shot prompts. It also includes provider selection logic:

- Gemini default model: `gemini-2.5-flash`
- Ollama fallback model in code: `qwen3:8b`

Optional agent dependencies are installed with:

```bash
pip install -e .[agent]
```



