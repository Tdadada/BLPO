# BLPO: Beyond-Local Policy Optimization for Long-Horizon LLM Agents

![BLPO overview](assets/method_overview.png)

BLPO improves long-horizon agent training by assigning credit at two complementary levels. A **local branch** compares normalized actions taken in the same concrete state. A **task-progress branch** shares evidence across trajectories whose histories imply the same progress condition, while context residualization, balanced EMA memory, and uncertainty-aware shrinkage limit biased cross-task comparisons. The two step-level signals are fused with the trajectory-relative advantage and used directly by the policy update; no critic is trained.

## Results

Mean ± standard deviation over three random evaluation seeds, as reported in the paper:

| Backbone | ALFWorld Seen success ↑ | ALFWorld Unseen success ↑ | WebShop success ↑ | WebShop task score ↑ |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | **94.01 ± 0.24** | **91.54 ± 0.60** | **74.02 ± 1.39** | **87.43 ± 0.56** |
| Qwen2.5-7B-Instruct | **97.14 ± 1.33** | **92.97 ± 0.64** | **79.43 ± 1.57** | **88.90 ± 1.26** |

## Repository layout

```text
blpo/
  algorithm.py             BLPO credit estimator and context-balanced memory
  keys.py                  local and task-progress state construction
  actions.py               concrete and abstract action normalization
  returns.py               trajectory and decision return utilities
  environments/            ALFWorld and WebShop rule implementations
  integration/trainer.py   minimal trainer adapter and memory checkpointing
  prompts/                 environment prompt builders
configs/                    reference configurations for both environments
docs/                       integration and prompt documentation
tests/                      deterministic unit tests
```

## Installation

BLPO is an algorithm package for on-policy agent training runtimes.

```bash
git clone https://github.com/Tdadada/BLPO.git
cd BLPO
pip install -e .
```

For audit logging to Parquet, install the optional dependency:

```bash
pip install -e ".[audit]"
```

## Environment setup

BLPO integrates with [verl-agent](https://github.com/langfengQ/verl-agent) and provides task abstractions for [ALFWorld](https://github.com/alfworld/alfworld) and [WebShop](https://github.com/princeton-nlp/WebShop). Set up the training runtime and the target environment before launching a run.

1. Install `verl-agent` and its CUDA, PyTorch, vLLM, Ray, and FSDP dependencies, then install BLPO in the same Python environment.
2. For ALFWorld, follow the [official ALFWorld setup](https://github.com/alfworld/alfworld). A typical text-environment setup is:

   ```bash
   pip install alfworld
   export ALFWORLD_DATA=/absolute/path/to/alfworld-data
   alfworld-download -f
   ```

3. For WebShop, follow the [official WebShop setup](https://github.com/princeton-nlp/WebShop). Its setup script downloads product/instruction data and builds the search index:

   ```bash
   git clone https://github.com/princeton-nlp/WebShop.git /absolute/path/to/webshop
   cd /absolute/path/to/webshop
   ./setup.sh -d all
   export WEBSHOP_HOME=/absolute/path/to/webshop
   ```

4. Download the Qwen2.5-Instruct backbone through a model provider supported by your runtime and set its path in the runtime configuration.

Set the model, task manifest, and environment paths in the host runtime configuration. The required per-decision metadata is listed in [docs/integration.md](docs/integration.md).

## Training integration

Create one estimator per training run and keep it alive across policy updates:

```python
from blpo import BLPOConfig
from blpo.integration import BLPOEstimator

estimator = BLPOEstimator(BLPOConfig(
    webshop_abstraction="decision_frontier_v4",  # ignored by ALFWorld
    role_current_batch_center=is_webshop,
))

advantages, returns, metrics, examples = estimator.compute(
    batch=rollout_batch,
    token_level_rewards=token_level_rewards,
    response_mask=response_mask,
    update=global_step,
    gamma=0.95,
)
rollout_batch.batch["advantages"] = advantages
rollout_batch.batch["returns"] = returns
```

Save and restore the BLPO memory together with the actor checkpoint:

```python
estimator.save(checkpoint_dir)
estimator.load(checkpoint_dir)
```

The complete batch contract and integration points are documented in [docs/integration.md](docs/integration.md). Reference settings are provided in [configs/alfworld.yaml](configs/alfworld.yaml) and [configs/webshop.yaml](configs/webshop.yaml).

## Reproducibility

Run the test suite with:

```bash
python -m pytest -q
```
