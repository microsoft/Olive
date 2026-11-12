# Provisional historical-review knowledge

Knowledge version: **0.2.0**. H01/H03 scope and exceptions unchanged. Source snapshot:
`ea6ae28111df93a82a0b04197751a57adba9d258`.
Recheck relevant behavior at the review's pinned revisions; this snapshot is not a
promise that future Olive versions retain every detail. Neither rule is approved policy.
These rules originate in historical reviews, corroborated by current code/tests.
The separate [architecture checks](architecture-rules.md) retain original IDs I06/I07.

## H01: capture CLI intent must reach the intended configuration

**Trigger:** changes to capture argument definitions, Model Builder recipe construction,
unset-value filtering, or option forwarding. Initial scope is the three boolean options
below in the Model Builder export path, not every Olive command or exporter.

Observed in `CaptureOnnxGraphCommand.register_subcommand` and `_get_run_config`,
`olive/cli/capture_onnx.py`:

| Input | Parsed value | Generated Model Builder pass option |
| --- | --- | --- |
| Omit `--exclude_embeds` or `--exclude_lm_head` | `False` | Explicit `False` |
| Omit `--enable_cuda_graph` | `None` | Unset |
| Supply any of these flags | `True` | `True` for that option |

The replacement loop skips only `None`. `ModelBuilder._run_for_config` in
`olive/passes/onnx/model_builder.py` forwards non-None options and then applies
`extra_options` overrides. `_check_extra_options` converts booleans to lowercase strings
before external validation; structured options can also undergo external normalization.
This proves Olive's forwarding boundary, not the external builder's default CUDA behavior.

**Finding threshold:** show a particular invocation and a changed mapping/default/filter
that loses or changes its intended option at a reachable consumer. Explain the consequence,
not just that a value changed. For example, `--exclude_lm_head` accidentally reading
`exclude_embeds` can request retention instead of exclusion. Dropping explicit false
values needs an actual downstream difference, not a generic warning about truthiness.

**Exceptions:** other exporters, Mobius, and metadata-only paths do not necessarily consume
these flags. `None` means unspecified, not disabled. There is no negative CUDA flag here.
User overrides and documented intentional interface changes may be valid. Do not demand
`store_true` universally or assume the external builder's defaults.

**Test evidence (inspected, not execution evidence):**
`test/cli/test_capture_onnx_args.py` checks exact omitted/present parser values; despite
its name, `test_capture_onnx_boolean_flags_default_to_disabled` asserts CUDA `None`.
`test/cli/test_cli.py::test_capture_onnx_command` checks mocked routing, not all three
option values. `test/passes/onnx/test_model_builder.py::test_model_builder_prechecks_extra_options`
checks mocked string forwarding, not real CUDA execution. None of these observations
establishes complete parser-to-pass matrix coverage.

**Enforcement:** deterministic parser/config assertions should carry established contracts;
AI guidance traces applicability, consumers, and exceptions. Missing assertions alone are
suggested tests, not findings.

**Smallest maintainer question:** must omitted `--enable_cuda_graph` continue leaving the
builder option unspecified rather than explicitly setting it to `False`?

**Public provenance:** human guidance and author qualification in
[PR #2647](https://github.com/microsoft/Olive/pull/2647), especially
[the default-preservation reply](https://github.com/microsoft/Olive/pull/2647#discussion_r3980916587).
The rule is a scoped interpretation, not a verbatim maintainer policy.

## H03: explicit CompositeModel selection

Architecture candidate **I04** overlaps this rule; review it here, not as a second check.

**Trigger:** changes to `ModelConfig.select_components`'s CompositeModel branch,
`CompositeModelHandler.select_components`, or callers relying on their selection behavior.
A composite here groups separately represented child models; it need not be multimodal.
Initial scope is explicit children, not directory discovery or package relocation.

Observed at the source snapshot:

- Selected children are copied, requested order is preserved, and one selected child is
  returned directly (unwrapped) rather than inside a one-child composite.
- Singleton parent/child attributes are merged shallowly with child keys winning.
  Multi-selection retains parent attributes; handler iteration applies inheritance.
- Handler selection rejects duplicate requests immediately. Config selection can return
  duplicate-name config; handler construction rejects duplicate names later.

Sources: `olive/model/config/model_config.py::ModelConfig.select_components`;
`olive/model/handler/composite.py::CompositeModelHandler.select_components`,
`__init__`, and `model_components`.

**Finding threshold:** identify a concrete selection that mutates a source child, changes
requested order, wrongly leaves a singleton wrapped, or replaces a child attribute with a
conflicting parent attribute. Cite changed code and the observed consequence. A duplicate
finding must show a violated boundary; the absence of an early config guard is not enough.
Do not require `deepcopy` specifically if another implementation preserves the behavior.

**Exceptions:** ordinary `model_components` iteration intentionally updates stored child
attributes; selection nonmutation does not forbid that API. No universal recursive parent
metadata isolation, file/weight copying, cache isolation, or concurrency guarantee follows.
Diffusers selection uses canonical order and remains a Diffusers config; Hugging Face
selection also has its own path. Do not apply Composite ordering/unwrapping rules to either.
Different duplicate-validation timing is not automatically a defect.

**Test evidence (all in `test/model/test_composite_model.py`, inspected only):**
`test_select_components_single_returns_unwrapped_child` checks identity/type, inherited
parent key, and source attributes remaining `None`, but not conflicting/nested keys.
`test_select_components_multiple_returns_subset_composite` checks requested order and
distinct identities. Handler empty/unknown/duplicate tests and constructor duplicate tests
do not establish eager config duplicate rejection.
`test_model_config_select_components_multiple_returns_composite_config` checks order/paths.
`test_model_config_select_components_single_inherits_parent_attributes` checks an actual
conflict with child precedence, but not source-config nonmutation.
`test_composite_model` checks iteration inheritance/serialization, not every singleton case.

**Enforcement:** primarily deterministic identity/order/type/attribute/error tests, with AI
guidance for applicability and intent. Source-config nonmutation and a conflicting-key
singleton handler case are useful targeted verification, not grounds for findings alone.

**Smallest maintainer question:** is duplicate rejection intentionally deferred from
`ModelConfig.select_components` until handler creation?

**Public provenance:** human-posted, explicitly AI-assisted
[review of PR #2480](https://github.com/microsoft/Olive/pull/2480#issuecomment-5358006743),
corroborated by
[the copying correction](https://github.com/microsoft/Olive/commit/ef2a4ee5ed04efbffa8fea57babc08becc6d674c)
and current selection assertions. These are evidence for provisional guidance, not approval.
