# Prompt construction

The prompt templates are defined in `blpo/prompts/`. Training and evaluation use the same construction path.

## Shared construction

At every environment decision, the runtime supplies the task, current textual observation, admissible actions, and the two most recent observation–action pairs. The complete environment prompt is passed as a single user message through the Qwen chat template with `add_generation_prompt=True`. Padding is applied on the left, and overlong inputs are also truncated from the left.

## ALFWorld

The initial observation already contains the natural-language task. The first prompt therefore includes the current observation and admissible actions. Later prompts additionally include the extracted task, total step count, and up to two recent observation–action pairs in the following form:

```text
[Observation k: '...', Action k: '...']
```

The reference configuration uses a 2,048-token prompt limit, a 512-token response limit, and at most 50 environment steps.

## WebShop

The runtime extracts the instruction and visible page text from the environment's `[SEP]`-delimited observation. It exposes `search[<query>]` when a search box is present and converts clickable page elements to `click[...]` actions. Later prompts add the same two-step history used in ALFWorld.

The reference configuration uses a 4,096-token prompt limit, a 512-token response limit, and at most 30 environment steps.
