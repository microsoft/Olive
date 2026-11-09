# Experimental Olive PR review Skill

A manually invoked, private advisory reviewer for H01 (capture CLI intent/configuration)
and H03 (explicit CompositeModel selection). Both rules are **provisional**, pending
maintainer feedback. No service, automatic workflow, approval, or merge capability.
This directory is portable Skill source, not a Python package.

## Try it safely

Use an operator-approved copy of this Skill outside the candidate PR checkout, or keep
the trusted checkout pinned while reading candidate revisions without checking them out.
Do not install instructions from a PR you are reviewing. No Olive dependencies are needed.
Existing Agency/GitHub read tools can supply PR metadata/diffs/pinned source.

### Copilot CLI

Copilot CLI 1.0.105 exposes these commands in installed help:

```powershell
copilot skill add --help
copilot skill list
```

Top-level `skills` is not automatically discovered. To register only this Skill, copy
the **complete** `olive-pr-review` directory into an operator-owned directory containing
no other Skills, then register that parent:

```powershell
copilot skill add "<trusted-skills-directory>"
copilot skill list
copilot
```

Confirm `olive-pr-review` appears. Start a fresh interactive session, then ask for it by
name using the prompts below; verify the load receipt. Keep normal trust/permission
prompts enabled. Directory registration persists in local CLI settings. To unregister
that custom directory later (not delete Skill files):

```powershell
copilot skill remove "<trusted-skills-directory>"
```

Passing only `SKILL.md` to `skill add` materializes that file, not its supporting files;
do not use that mode for this Skill. Do not register all `skills/dev`: that also exposes
unrelated, write-capable workflows.

**Explicit loading fallback:** in an existing approved Agency/Copilot session, ask the
assistant to read the three trusted files by full path, then follow them. This is explicit
instruction loading, not proof of runtime auto-discovery. `README.md` is optional teaching
material; the required files are `SKILL.md`, `references/rules.md`, and
`references/review-template.md`. Require actual file reads and a hash/version receipt.
Do not invent a `/olive-pr-review` slash command.

### Invocation prompts

Replace placeholders; output must be outside the Olive checkout. Bare PR numbers are
always interpreted as `microsoft/Olive`. Do not paste credentials.

```text
Use the trusted olive-pr-review Skill. Read SKILL.md, references/rules.md, and
references/review-template.md from <approved-skill-directory> in full first.
Review microsoft/Olive PR <number> (or https://github.com/microsoft/Olive/pull/<number>).
Use approved GitHub reads only; do not check out or execute candidate code.
Save the private report to <approved-output-file-outside-checkout>.
Record base/head/comparison identities and the trusted-file load receipt.
```

```text
Explicitly load and follow SKILL.md, references/rules.md, and
references/review-template.md from <approved-skill-directory>.
Review the two-endpoint diff in <Olive-repository> from <full-base-commit>
to <full-head-commit>. Read pinned source/tests, not dirty worktree content.
Save the private report to <approved-output-file-outside-checkout>.
Do not execute candidate code or perform GitHub writes.
```

The playbook asks the assistant to collect and reason; it is not a deterministic bug
detector or a tool-permission sandbox. If required reads/identities are unavailable, it
must report insufficient evidence. No supported finding is never an approval.

## Olive versus this reviewer

Olive optimizes AI models for deployment. **Configuration** describes the input model,
transformations, target, and optional measurements. A **workflow** resolves that recipe.
A **model handler** represents a model and its resources/loading behavior; a **pass**
transforms it. The **engine** coordinates passes, caching, search, and configured
**evaluation** (measurement of metrics such as accuracy or latency). Not every workflow
uses search or evaluates quality.

This Skill reviews developer changes to those mechanisms. It neither runs optimization
nor measures model quality. Its knowledge helps an assistant recognize specific
configuration/object-boundary mistakes; maintainers still decide intended contracts.

## H01 walkthrough: CLI input to a pass option

At source snapshot `ea6ae28111df93a82a0b04197751a57adba9d258`, consider:

```text
olive capture-onnx-graph --model_name_or_path <model> --use_model_builder --enable_cuda_graph
```

This is an explanatory input, not a command to execute during review.

1. `olive/cli/capture_onnx.py::CaptureOnnxGraphCommand.register_subcommand` defines
   `enable_cuda_graph` as a presence flag. Present means `True`; omitted means `None`.
   **Unset** means the user did not specify a value, distinct from explicitly disabled.
2. `_get_run_config` chooses the Model Builder export branch and maps that parsed value
   into `passes.m.enable_cuda_graph`. Its replacement loop skips only `None`.
   Thus omission leaves the option out; presence writes `True`. The exclusion flags
   instead default to `False` and are explicitly written.
3. `olive/passes/onnx/model_builder.py::ModelBuilder._run_for_config` forwards non-None
   options, then lets user `extra_options` override them. `_check_extra_options` converts
   booleans to `"true"`/`"false"` before external builder validation.

The CLI owns parsing and recipe creation; the pass owns translation to the external
builder. A parser-only test cannot prove that the correct value crosses both boundaries.
For example, mapping `exclude_lm_head` from `exclude_embeds` would make
`--exclude_lm_head` alone forward `False`, requesting retention instead of exclusion.

These are observed source mechanics. Whether omission must remain unspecified forever
is a maintainer decision. Neither `None` nor these source reads establishes the external
builder's default CUDA behavior. See `test/cli/test_capture_onnx_args.py` for parser
assertions and the rule reference for their limits.

## H03 walkthrough: select one named model

Imagine an explicit CompositeModel with two separately represented ONNX children named
`encoder` and `decoder`. This could be a text-only model; **composite** describes grouping,
not whether the model accepts text and images (**multimodal**).

1. A caller supplies `names=["decoder"]` to
   `olive/model/config/model_config.py::ModelConfig.select_components`. The config says
   which children/resources exist; no model inference is needed to choose one.
2. In the CompositeModel branch, the selector maps names to children, copies the selected
   child, and returns its child config directly. **Unwrapping** means returning the decoder
   rather than a composite wrapper containing only that decoder.
3. Shared parent attributes are merged into the selected child. If parent says
   `{"shared": "yes", "role": "parent"}` and child says `{"role": "decoder"}`, the result
   retains `shared` and `role="decoder"`. **Child precedence** means the child's value wins
   for a conflicting key. This is a shallow merge, not a recursive isolation guarantee.
4. `ModelConfig.create_model` constructs the appropriate handler. An already constructed
   `CompositeModelHandler` also has `select_components`, with the same copying/unwrapping
   behavior. Selecting `["decoder", "encoder"]` instead returns a composite in that order.

Removing the copy in handler selection would let singleton attribute inheritance modify
the original child. A later selection could then see context from a previous operation.
Identity/source-attribute assertions in `test/model/test_composite_model.py` support this
concern without executing a model.

Do not mistake validation timing for that bug: the handler selector rejects duplicates
immediately, while the config selector can defer rejection until handler construction.
Also, ordinary handler component iteration intentionally updates child attributes.
Diffusers and Hugging Face selection have different paths. Observed Composite behavior
is not a universal model-selection design promise; intended duplicate timing remains a
maintainer question.

## Validation scope

Development demonstrations can use explicitly synthetic, content-addressed proposed
source/diffs anchored to an actual Olive base commit. Keep expected answers separate and
out of reviewer input; retain private reports outside the contribution. Include a supported
issue and a valid/out-of-scope case, and record which instructions/references were loaded.
Do not execute historical/candidate code or inspect reserved evaluation material.

Fixture success demonstrates loading, applicability, and output handling, not benchmark
accuracy, autonomous approval readiness, or reduced review effort. A next evaluation
should compare the same new examples/settings with and without this knowledge, followed
by human adjudication of useful findings, misses, and noise.
