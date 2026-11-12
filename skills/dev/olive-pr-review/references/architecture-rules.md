# Provisional architecture-derived review guidance

Knowledge version: **0.2.0**. These checks originate in architecture candidates **I06**
and **I07**, not the historical PR study. The original IDs are retained for traceability.
Both are experimental advice, not maintainer-approved contracts or approval requirements.

The architecture investigation inspected `1647eb2b589ae8a82b2dbe4429e5fe4335c40092`.
This increment rechecked the relevant source/tests at
`93de1da4251363e3875f4d02161cab0aacaece13` (prototype branch; runtime source unchanged
from base `ea6ae28111df93a82a0b04197751a57adba9d258`). The inspected engine, cache,
base Pass, and engine tests also match the original investigation revision. Recheck
behavior at each reviewed PR's pinned revisions. Existing behavior is evidence, not proof
of permanent intent. Referenced Olive tests were inspected, **not executed**.

## I06: a saved pass result must not bypass today's validation

**Plain-language scenario:** a transformation can be reusable across devices, but a
particular request can still be invalid. Finding a saved output must not turn a rejected
request into a successful one.

**Trigger:** changes to `Engine._run_pass`, pass validation, accelerator-agnostic reuse,
or an early return from the run cache. This is the transformation cache, not metric reuse.
Sources:
[Engine._run_pass](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/olive/engine/engine.py#L655-L758),
[Pass validation/reuse](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/olive/passes/olive_pass.py),
and [OliveCache](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/olive/cache.py#L234-L313).

**Observed path:** `_run_pass` imports the registered pass class, calls
`validate_config(config, accelerator_spec)`, and returns `INVALID_CONFIG, None` on
rejection. Only then does it construct the pass, calculate an output ID, and load a cached
run/model. For accelerator-agnostic computation, `run_accel=None` intentionally excludes
the accelerator from the run identity. A usable cached model returns before host execution.

**Review procedure:**

1. Follow the actual validator and every cache-hit return in the changed path. Identify
   the configuration and current accelerator for which validation rejects the request.
2. Show that a usable cache entry can exist under the resulting ID, and trace the changed
   branch that returns it without equivalent validation. Check whether another guard
   already enforces the same condition before that return.
3. Compare cold-cache and warm-cache behavior. Cite the changed line, validator result,
   cache identity, and observable result. A hypothetical ordering concern alone is a question.

**Finding threshold:** establish a reachable rejected request plus a usable cache hit
that newly returns a model instead of the rejection. For example, a registered
accelerator-agnostic pass can reuse a CPU-produced artifact while its validator rejects
the same configuration on another accelerator. Identify that pass/validator, or explicitly
label a supplied extension/mocked scenario; do not pretend a synthetic pass is built in.

**Exceptions and valid counterexamples:** sharing computation across accelerators is
intentional, not itself a defect. A refactor retaining equivalent validation before all
successful returns is valid; validation need not remain on the same line or in the same
function. A validator returning True for all affected requests cannot support a rejection
finding. Missing or unloadable cached models fall through to execution. Direct Pass APIs
are not automatically governed by this engine boundary. Do not claim validation is pure:
some validators normalize configuration. Do not infer universal cache/environment validity.

**Test evidence and gaps:**
[TestEngine.test_pass_cache](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/test/engine/test_engine.py#L548-L607)
patches conversion to be accelerator-specific and asserts one host pass call in one run.
Despite its name, it does **not** establish a second-run hit or rejection-before-hit.
The inspected base `Pass.validate_config` returns True; subclasses can reject requests.
No focused warm-cache/rejected-request assertion was established in this bounded check.
Missing coverage is a suggested test, not a bug. Current ordering and the extension API
support this narrow advisory check, not a universal tested guarantee.

**Enforcement:** both. Prefer a deterministic mocked-engine regression with a usable
cache entry and a rejecting validator: expect `INVALID_CONFIG` and no cached-success
return/host execution, plus a valid-hit control. AI review traces reachability and
equivalent guards. Synthetic text demonstrations do not execute that regression.

**Maintainer question:** should the engine always reject a request when its current
`validate_config` rejects it, even when an accelerator-agnostic output already exists?

## I07: changed evaluation requests must not silently reuse old measurements

**Plain-language scenario:** measuring accuracy and measuring latency are different
requests, even for the same model on the same device. Returning yesterday's accuracy
result for today's latency request is not a successful evaluation.

**Trigger:** changes to `Engine._evaluate_model`, evaluator serialization, evaluation
key construction, or cache read/write plumbing. This is the metric cache, distinct from I06.
Sources:
[Engine._evaluate_model](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/olive/engine/engine.py#L760-L838)
and [evaluation storage](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/olive/cache.py#L283-L294).

**Observed path:** the engine combines model/accelerator identity with the first eight
characters of `hash_dict(evaluator_config.to_json())` when an evaluator exists. It uses
that key for both `_load_evaluation` and `_cache_evaluation`. A non-None cached signal
returns without calling `target.evaluate_model`; otherwise the target measures and stores
the new result. Footprint records deliberately retain the original model ID, not this key.

**Review procedure:**

1. Hold model, accelerator, and cache namespace constant. Identify two concrete evaluator
   requests whose distinction matters: metric, backend, task, or serialized parameter.
2. Trace their serialized identities through cache reads **and** writes. Show the changed
   code aliases them and permits the first signal to satisfy the second request.
3. Check equivalent normalization, separate namespaces, disabled reuse, migration, and
   intentional contract changes. Explain the stale/wrong result or skipped required
   reevaluation; merely obtaining a different hash is not an adverse effect.

**Finding threshold:** a concrete non-equivalent request newly receives a previous
request's cached metrics, or is incorrectly prevented from obtaining its own measurement.
Cite the changed key/serialization/plumbing and reachable cache hit. Do not demand this
exact hashing algorithm or eight-character format if another identity preserves semantics.

**Exceptions and valid counterexamples:** extracting a local hash variable while using
the same key for loads/saves is valid. Identical or established equivalent requests may
reuse results. `evaluator_config=None` has no evaluator hash in the inspected code; do not
declare that branch defective without a reachable adverse case. The accelerator suffix
and original footprint ID have separate purposes. External dataset contents, script
contents, runtime versions, and all hardware state are not proven to be fingerprinted.
Do not flag these pre-existing limitations as introduced regressions or promise complete
reproducibility. Hash truncation alone is not a demonstrated collision.

**Test evidence:**
[TestEngine.test_evaluate_model_cache_key_includes_evaluator_config](https://github.com/microsoft/Olive/blob/93de1da4251363e3875f4d02161cab0aacaece13/test/engine/test_engine.py#L449-L495)
holds model/accelerator fixed and varies the accuracy goal (0.99 versus 0.5). Its mocked
target call count is 1 after the first request, remains 1 for an identical request, and
becomes 2 for the changed config. It does not assert different numerical outputs, run a
real evaluator, or cover every backend/task/data/version change.

**Enforcement:** both, primarily deterministic cache reuse/separation tests using a mocked
target, with AI review following serialized fields, namespaces, and intentional equivalence.
Missing tests alone belong under suggested verification.

**Maintainer question:** which evaluator differences are intentionally equivalent for
metric reuse, and must distinct serialized configurations always trigger reevaluation?
