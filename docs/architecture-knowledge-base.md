# Olive architecture knowledge base

## Status, scope, and evidence

**Initial architecture knowledge base for experienced-engineer review; not approved engineering policy.**

This document consolidates the T3 architecture investigation into one reviewable reference. It explains the current implementation, highlights consequential boundaries, and proposes questions and candidate preservation requirements. It does not establish approval criteria or authorize a review agent.

| Field | Basis |
| --- | --- |
| Repository | microsoft/Olive |
| Source revision | `1647eb2b589ae8a82b2dbe4429e5fe4335c40092` |
| Source checkout | `C:\Users\nbudnarain\Olive`; clean `main` when verified |
| Existing reference | `docs\architecture.md:1-373` |
| Method | Consolidation of static source/documentation inspection and selected test assertions from T3 |
| Runtime evidence | Tests, optimization workflows, model downloads, numerical checks, and hardware runs were not performed |
| Maintainer endorsement | Pending; no candidate invariant is approved |

**Observed** means behavior supported by inspected implementation. **Inference** means an interpretation or proposed requirement, not established intent. **Open** means a contract or evidence boundary requiring confirmation. References are relative to the Olive repository at the pinned revision; cited tests were inspected, not run.

Reviewers should concentrate on sections 6-9 after reading the architectural overview. The detailed research remains available for traceability in section 10.

## 1. Architectural overview

**Observed:** Olive orchestrates model transformations, execution environments, and evaluation/search. ONNX Runtime is central to device/provider integration, but Olive is not restricted to ONNX graphs: registered handlers and backend-specific passes support other model representations.

The central separation is between:

- **Workflow configuration:** what model, transformations, data, environments, and evaluation to use.
- **Engine orchestration:** how to execute those transformations, reuse results, and track candidates.
- **Backend operations:** how a particular handler, pass, evaluator, or system implements the work.
- **Artifacts and results:** the resulting model resources, lineage, metrics, and optional assembled package.

```text
CLI options or Python workflow configuration
                    |
           parse / resolve configuration
                    |
          +---------+-----------------------+
          |                                 |
     ordinary run                      named builds
          |                         prevalidate all builds
          |                         bounded parallel/serial runs
          +---------------+-----------------+
                          |
                     _run_single
                          |
             +------------+--------------+
             |                           |
        Docker host                 other supported hosts
    whole-workflow container             |
                                  run_engine / Engine
                                         |
                              fixed pass chain or search trials
                                         |
                           pass host (or target) transforms model
                                         |
                            target evaluates when configured
                                         |
                             cache + lineage + WorkflowOutput
                                         |
                      named component builds may assemble a package
```

The diagram summarizes inspected control flow, not a dynamic trace. Named builds have additional restrictions described below; Docker dispatch does not imply Docker-hosted named builds are supported.

Evidence: `run_engine`, `run`, and `_run_single`, `olive\workflows\run\run.py:101-253`; `parse_run_config`, `olive\workflows\run\builds.py:1-240`; `Engine`, `olive\engine\engine.py:612-838`.

## 2. Major components and responsibilities

| Component | Responsibility and important boundary | Source |
| --- | --- | --- |
| CLI | Converts user options into workflow configuration; some commands generate ordered optimization recipes rather than simply forward JSON. | `OptimizeCommand`, `olive\cli\optimize.py:224-252`; `WorkflowRunCommand`, `olive\cli\run.py:23-104` |
| Workflow/configuration | Resolves named systems, evaluators, data, and passes; expands builds and dispatches execution/assembly. | `RunConfig`, `olive\workflows\run\config.py:245-465`; `olive\workflows\run\builds.py:1-240` |
| Engine | Executes pass groups or search samples, validates applicability, consults caches, routes work, evaluates, and records lineage. | `Engine`, `olive\engine\engine.py:154-297`; `olive\engine\engine.py:612-838` |
| Passes | Define configurable transformations and their fixed/searchable parameters. Backend implementations supply the transformation hook. | `Pass`, `olive\passes\olive_pass.py:41-195`; `olive\passes\olive_pass.py:467-471` |
| Models/resources | Separate model configuration and resource identity from loading, inference sessions, and serialization. | `ModelConfig`, `olive\model\config\model_config.py:21-100`; `OliveModelHandler`, `olive\model\handler\base.py:15-105` |
| Data/evaluators | Build input pipelines, run configured metrics, and normalize metric results for engine/search use. | `DataContainer`, `olive\data\container\data_container.py:15-60`; `_OliveEvaluator.evaluate`, `olive\evaluator\olive_evaluator.py:496-531` |
| Search | Represents choices and conditional spaces, yields trial samples, records feedback, and traverses ranked results. | `SearchSpace`, `olive\search\search_space.py:94-210`; `SearchStrategy`, `olive\search\search_strategy.py:212-349` |
| Systems/hardware | Distinguishes the transformation host from the evaluation target and carries device/provider context. System capabilities are not uniform. | `OliveSystem`, `olive\systems\olive_system.py:23-49`; `AcceleratorSpec`, `olive\hardware\accelerator.py:28-70` |
| Cache/output/assembly | Materializes resources, stores run/evaluation results, exposes candidate outputs, and assembles supported component packages. | `olive\cache.py:120-225`; `WorkflowOutput`, `olive\engine\output.py:163-259`; `olive\workflows\run\component_assembly.py:55-147` |

These are responsibilities, not independent services. Most ordinary orchestration occurs within the Python workflow process; selected systems introduce subprocess/container boundaries.

## 3. Important execution flows

### Configuration and fixed execution

**Observed:** Workflow validation resolves named references and rejects invalid combinations. Search requires an evaluator; unresolved searchable placeholders are rejected when search is disabled. Pass packages can be imported dynamically. The engine executes the first configured alternative in each pass group during fixed execution.

For each pass, the engine validates applicability before consulting its run cache, chooses the configured pass host/engine host or target, and materializes required resources. The pass returns a model configuration through the system boundary. Final-stage evaluation is conditional on evaluator configuration; there is no promise that every intermediate model is evaluated.

Evidence: `olive\workflows\run\config.py:245-465`; `olive\package_config.py:46-58`; `olive\engine\engine.py:350-390`; `olive\engine\engine.py:612-758`.

**Observed:** `OptimizeCommand` constructs an ordered, conditional conversion/optimization/quantization pipeline. The shared CLI dry-run path saves generated configuration and returns before invoking the workflow. Preparation can still create directories or otherwise execute configuration-generation logic.

**Inference:** Dry-run should be described as a way to inspect generated intent, not proof of complete workflow validation, numerical correctness, provider availability, offline operation, or absence of side effects.

Evidence: `olive\cli\optimize.py:320-412`; `BaseOliveCLICommand._run_workflow`, `olive\cli\base.py:287-358`.

### Search and feedback

**Observed:** `SearchStrategy` is iterable and yields `SearchSample` objects containing a search point and input-model IDs. Feedback updates result and sampler state. Pass-by-pass traversal can revisit earlier ranked results rather than retaining only a single best predecessor.

`SearchResults` applies objective priority/direction and lexicographic ordering. Footprint Pareto selection is a separate mechanism. Goal filtering can fall back to ranking without goals; `meets_goals` does not independently establish completeness of all expected metrics.

**Inference:** Trial lineage, ranking, goal attainment, and metric completeness should be explained separately. A selected candidate is not necessarily a candidate that met every requested goal.

Evidence: `olive\search\search_strategy.py:212-349`; `olive\search\search_results.py:15-115`; `olive\engine\footprint.py:118-135`; `olive\engine\engine.py:412-458`.

### Named builds and package assembly

**Observed:** Build expansion prevalidates all configurations before launching engines. Named builds currently require LocalSystem engine hosts. Execution uses bounded threads, per-build package-config copies, and context-local cache settings; a configured non-thread-safe pass forces serial execution. Results retain configured build-name order, and build failures are aggregated. Missing/no-output engine results are rejected.

Component builds additionally validate write ownership and source/output separation. Successful component transformations may then feed format-specific ONNX or Hugging Face assembly; this is not generic concatenation of arbitrary model outputs.

Evidence: `olive\workflows\run\builds.py:1-240`; `olive\workflows\run\run.py:174-260`; `olive\cache.py:25-95`.

## 4. Contracts that matter across components

### Models, serialization, and composite behavior

**Observed:** `ModelConfig` selects a registered handler. Handlers expose resources, loading/session operations, I/O, and JSON representation, but not every handler implements every operation. Resources can be local files/folders, names, or Azure ML registry references; recognizing a resource description does not prove availability.

Live functions/objects can have identity-oriented serialized representations without being reconstructable from JSON. Python-environment runners require reconstructable configuration. This differs from supported in-memory use of local objects.

Composite selection copies children, preserves requested order, unwraps a single selection, and inherits parent attributes with **child precedence**. That is distinct from ordinary `Pass.run`, which uses **output-or-input attribute fallback**, not a universal dictionary merge.

Evidence: `olive\model\config\model_config.py:21-47`; `olive\resource_path.py:29-158`; `olive\common\config_utils.py:28-110`; `olive\model\handler\composite.py:175-252`; `olive\passes\olive_pass.py:225-304`.

### Pass lifecycle and artifacts

**Observed:** Pass subclasses register under lowercase class names and construct accelerator-aware configuration. `get_config_class` returns a generated config class together with default metadata. `Pass.run` initializes lazily, handles distributed ranks, and recurses over composite children unless the pass handles composites directly.

Additional files/directories are carried forward separately from ordinary attributes. Paths are rebased, and newly generated output files are not overwritten by colliding input sidecars. Missing/unsupported resource cases can warn and return; this is not unconditional copying of every resource.

Evidence: `olive\passes\olive_pass.py:41-195`; `olive\passes\olive_pass.py:206-304`; supporting assertions in `test\passes\test_pass.py:13-100`.

### Cache identity, evaluation, and success

**Observed:** Model identity combines normalized configuration with resource-dependent identity. Local files, Hugging Face references, Azure resources, and callables take different paths. Evaluation cache identity includes model, accelerator context, and serialized evaluator configuration. Pass applicability validation precedes run-cache reuse.

**Inference:** These identities support reuse but are not a complete reproducibility guarantee across dependency versions, external mutable data, user-script behavior, or hardware.

**Observed:** Evaluators dispatch configured accuracy, latency, throughput, size, and custom metrics. ONNX metric inference settings override model settings without mutating the originals in inspected tests. Some engine failures can result in no output after logging, while search may prune failed trials. Multi-build orchestration explicitly rejects no-output builds.

**Inference:** Output existence, successful execution, measured quality, goals passed, and hardware validation are separate claims. A `WorkflowOutput` alone establishes none of them universally.

Evidence: `olive\model\config\model_config.py:252-315`; `olive\engine\engine.py:658-838`; `olive\evaluator\olive_evaluator.py:496-549`; `olive\engine\engine.py:242-297`; `olive\engine\output.py:163-259`.

### Systems and provider evidence

**Observed:** The system pass boundary exchanges `ModelConfig`; evaluation receives evaluator configuration plus `AcceleratorSpec` and returns `MetricResult`. LocalSystem runs these operations locally. PythonEnvironmentSystem serializes configurations, invokes checked subprocess runners, and reads their JSON outputs.

Docker instead runs an entire workflow; direct Docker pass/evaluation methods are unimplemented. Accelerator normalization can omit runtime-provider checks when target execution is unnecessary. The inspected accelerator creation path selects one normalized device/provider.

**Inference:** Configured provider intent, declared pass support, installed provider availability, and successful target execution are distinct evidence levels. Dependency-environment separation is not a security sandbox.

Evidence: `olive\systems\olive_system.py:23-49`; `olive\systems\local.py:20-66`; `olive\systems\python_environment\python_environment_system.py:35-138`; `olive\systems\docker\docker_system.py:305-331`; `olive\workflows\run\run.py:247-253`; `olive\systems\accelerator_creator.py:18-201`.

### Data and extension configuration

**Observed:** Data preparation is load -> preprocess -> dataloader; postprocessing applies separately to model output. Configuration resolves registered defaults and can import user scripts during validation. Custom evaluator configuration and pass imports also introduce executable extension boundaries.

Registry behavior is not uniform: model-handler duplicates raise, while differing data/evaluator registrations can log and replace definitions. There is no source-backed universal duplicate-registration rule.

Evidence: `olive\data\config.py:44-223`; `olive\data\container\data_container.py:15-60`; `olive\evaluator\olive_evaluator.py:2488-2570`; `olive\model\config\registry.py:8-38`; `olive\data\registry.py:28-66`; `olive\evaluator\registry.py:18-55`.

## 5. Component packages and operational boundaries

**Observed - ONNX assembly:** Assembly stages replacements, preserves unbuilt components and their referenced assets, confines external-data/context paths, rebases supported collisions, and rejects conflicting updates. Publication includes destination checks and rollback tracking.

**Observed - Hugging Face assembly:** Assembly works with component-scoped safetensors, validates normalized configurations and unowned tensor names/shapes/dtypes, handles shared/tied weights, merges quantization metadata, and publishes checkpoint artifacts.

**Open:** Neither staging nor rollback establishes universal crash-safe atomic publication. HF unowned-tensor metadata agreement is not universal tensor-value equality. Supported combinations, ownership, portability after relocation, and publication guarantees need explicit maintainer confirmation.

Evidence: `olive\workflows\run\component_assembly.py:231-415`; `olive\workflows\run\component_assembly.py:513-690`; `_validate_build_compatibility`, `olive\workflows\run\hf_component_assembly.py:443-503`; `try_assemble_hf_component_builds`, `olive\workflows\run\hf_component_assembly.py:1021-1119`.

**Observed - telemetry:** Telemetry is best-effort, suppresses its own failures, disables logging in detected CI, and caches failed payloads in encoded line-file entries. An older SQLite-queue description does not match this revision. Consent, retention, and acceptable event-loss behavior remain owner questions.

Evidence: `olive\telemetry\telemetry.py:255-342`; `olive\telemetry\telemetry.py:437-625`.

**Observed - tests/CI:** GitHub lint and model smoke workflows and Azure Linux/Windows CPU jobs cover different scopes. GPU jobs are commented out in the inspected primary Azure pipeline. A separate shell runner can return success after timeout if XML exists; that is not the CPU template's direct coverage command. Root pytest fixtures import optional ML packages and create assets even for small selectors.

**Inference:** Later review validation needs relevant, complete results for the reviewed revision; generic green status is not a substitute for provider/numerical coverage. This is a proposed evidence practice, not a new branch requirement.

Evidence: `.github\workflows\lint.yml:41-72`; `.github\workflows\test-model-fast.yml:5-57`; `.azure_pipelines\olive-ci.yaml:1-58`; `.azure_pipelines\job_templates\olive-test-cpu-template.yaml:48-73`; `.azure_pipelines\scripts\run_test.sh:57-79`; `test\conftest.py:15-47`.

## 6. Architectural choices and supported rationale

The choices below are visible in implementation. Their consequences are source-backed; the rationale column is explicitly **inferred**, not a claim about historical author intent.

| Observed architectural choice | Consequence | Inferred rationale / tradeoff to validate |
| --- | --- | --- |
| Registered handlers/passes plus configuration-mediated execution | Orchestration can select backend implementations without encoding all backend behavior in the engine. | Supports extension and environment changes, at the cost of schema, registry, and serialization compatibility concerns. |
| Separate transformation host and evaluation target | A transformation need not execute on the hardware that eventually runs inference. | Enables cross-target workflows; provider availability checks cannot be unconditional at every stage. |
| Distinct run/evaluation caches and explicit trial lineage | Transformation reuse, measurement reuse, and candidate provenance have different identities. | Avoids conflating model generation with evaluation, but requires clear cache-equivalence assumptions. |
| Prevalidated named builds with capability-driven serial fallback | Configuration errors are caught before engines launch; unsafe pass metadata reduces concurrency. | Balances throughput with shared-process constraints; metadata alone cannot prove race freedom. |
| Format-specific assembly and component ownership checks | ONNX assets and HF tensors receive different compatibility/preservation handling. | Protects untouched model content while avoiding an unjustified generic merge contract. |

Source basis: sections 2-5. **Open:** Which of these are stable public design commitments, and which are replaceable implementation strategies?

## 7. Candidate architectural invariants for future review

**Every row is an inferred preservation proposal, not approved policy.** Archive IDs retain traceability to the full candidate catalog. The corresponding source mechanisms are explained above; changing current behavior can be valid if maintainers deliberately change its contract.

| Archive IDs | Candidate preservation requirement | Scope / caution |
| --- | --- | --- |
| I01-I02 | Resolve configuration references and fixed/search requirements before executing the affected workflow/build. | Parsing may execute imports; this is not a side-effect-free guarantee for every direct API. |
| I03 | Reject unreconstructable live-object payloads at JSON/subprocess boundaries rather than pretending they round-trip. | Do not forbid supported local in-memory objects. |
| I04 | Preserve composite selection order, source nonmutation, single-child unwrapping, and child-preferred attributes. | Do not extrapolate to ordinary pass attribute merging or require homogeneous child types without confirmation. |
| I05 | Preserve required sidecars without replacing newly generated output assets with stale input copies. | Existing missing/unsupported-resource warnings remain relevant exceptions. |
| I06-I07 | Keep applicability validation ahead of cache reuse and distinguish evaluation configurations in cache identity. | This does not imply complete environment/data reproducibility. |
| I08 | Preserve ONNX metric-setting precedence and avoid mutating source settings during evaluation preparation. | Other backends may have different settings contracts. |
| I09-I10 | Retain host/target distinctions and report actual system/provider evidence accurately. | Docker whole-workflow support is not direct-operation support; offline transformation can be valid. |
| I11 | Keep search configurations, input-model lineage, pruning, and feedback associated with the correct trial. | Selected output need not meet all goals; missing-metric policy is unresolved. |
| I12 | Bound build concurrency, respect serial-only pass metadata, and isolate per-build mutable/cache context. | Threads and config copies do not establish all-state isolation. |
| I13-I14 | Keep component writes away from source packages and preserve unbuilt/shared assets within accepted ownership/path boundaries. | Collision handling is format-specific; publication is not universally atomic. |
| I15 | Reject incompatible HF component configurations and preserve shared-weight/quantization ownership during assembly. | Metadata checks are not universal tensor-value comparisons. |
| I16 | Distinguish missing output, failed execution, unevaluated output, unmet goals, and validated success. | Existing ordinary-workflow/search/build error semantics differ and must not be silently redefined. |

Selected supporting test evidence:

- Composite selection and precedence: `test\model\test_composite_model.py:78-158`; pass sidecar/content preservation: `test\passes\test_pass.py:13-100`.
- Evaluator-config cache identity: `test_evaluate_model_cache_key_includes_evaluator_config`, `test\engine\test_engine.py:449-495`; ONNX settings/nonmutation: `test\evaluator\test_olive_evaluator.py:385-413`.
- Build prevalidation/host restrictions: `test\workflows\test_run_builds.py:377-430`; bounded concurrency: `test\workflows\test_run_builds.py:455-521`; empty-output rejection: `test\workflows\test_run_builds.py:289-301`.
- Source/write ownership checks: `test\workflows\test_run_config_builds.py:191-245`; ONNX preservation/confinement/conflicts: `test\workflows\test_component_assembly.py:252-307` and `test\workflows\test_component_assembly.py:978-1080`.
- HF compatibility rejection: `test_build_compatibility_rejects_config_and_unoptimized_tensor_mismatches`, `test\workflows\test_hf_component_assembly.py:906-946`.

These are supporting examples, not exhaustive proofs. T3 did not inspect a focused assertion for every candidate, notably serialization rejection, applicability-before-cache ordering, and provider normalization; the archived catalog records those gaps.

## 8. Architectural gaps and uncertainties

**Observed documentation gaps:** The existing architecture document's broad decomposition remains useful, but its SearchSpace/SearchResults/SearchStrategy APIs and System signatures are stale. Its Pass config-class return contract is incomplete, callable portability wording is outdated, and a DataConfig example has inconsistent registration/selection names. Coverage is insufficient for current model/resource identity, evaluator semantics, CLI generation, named builds, assembly, and operations.

**Open intended contracts:** Source cannot establish compatibility promises, required numerical tolerances, representative calibration/benchmark datasets, complete hardware support, cache invalidation guarantees, composite type restrictions, or crash/concurrent-publication guarantees. Successful output and complete quality evidence are not interchangeable.

**Coverage limits:** T3 did not exhaustively audit every pass/quantizer, specialized evaluator, vendor SDK, packaging/export branch, dependency combination, or telemetry event. It obtained no runtime trace, resolved repository-wide graph, live ruleset validation, or hardware results. Source citations refer to the pinned revision, not ongoing branch behavior.

The archived gap analysis contains G01-G20 with detailed comparisons. Correcting official documentation is a separate future action; this document does not modify it.

## 9. Concise maintainer-validation agenda

The manager and experienced engineer can review this as one packet and route specialized questions as needed. Archive question IDs are retained for detail, not as separate mandatory checklists.

1. **Public contracts (Q01-Q03):** Which Python APIs, config/JSON formats, CLI defaults, extension patterns, and validation/dry-run semantics are stable? Which architecture reference should become canonical?
2. **Quality and target evidence (Q04, Q09, Q14):** Which changes require numerical/calibration checks, actual provider execution, and specific CI coverage? Who sets tolerances, and what counts as complete evidence?
3. **Search and status (Q05-Q06):** Is fallback without satisfied goals intentional? What happens with missing metrics? How should callers distinguish pruning, execution failure, no output, and successful-but-unevaluated output?
4. **Cache equivalence (Q07):** Which dependency, data, script, provider, or hardware changes intentionally invalidate results? What reproducibility promises are explicitly outside cache identity?
5. **Artifact and assembly contracts (Q08, Q10):** What composite types, attribute precedence, resource ownership, relocation/reload behavior, shared-weight checks, and publication guarantees are supported?
6. **Concurrency and environments (Q11-Q12):** What evidence permits a pass to be marked thread-safe? Which system operations are supported, and what separate infrastructure would be needed for future untrusted-code validation?
7. **Telemetry (Q13):** What consent/disable, retention, event filtering, and event-loss behavior should the architecture document promise?

**Proposed validation process (Inference):** Record corrections to observed facts separately from decisions about intended contracts. For each candidate, record accept, accept with scope/exceptions, reject, or unresolved, with rationale and applicable revision. No such dispositions are recorded yet. User review of T3 does not by itself approve these as engineering policy.

## 10. Provenance and persistent workspace

Project workspace: `C:\Users\nbudnarain\olive-review-agent`.

This consolidated document: `knowledge\olive-architecture-knowledge-base.md`.

The four T3 artifacts are preserved byte-for-byte under `knowledge\research\`:

| Research artifact | Detailed material retained |
| --- | --- |
| `research\olive-architecture-draft.md` | Fifteen architecture sections and extended evidence |
| `research\olive-architecture-gap-analysis.md` | G01-G20 documentation comparisons and omissions |
| `research\architectural-invariants-candidates.md` | I01-I16 mechanisms, proposals, scope/exceptions, and test gaps |
| `research\architecture-validation-questions.md` | Q01-Q14 and the original pending decision log |

Paths in the table are relative to this document's `knowledge` directory. The original copies also remain untouched in the session workspace at `C:\Users\nbudnarain\.copilot\session-state\82a82492-fb55-4110-ad27-c4bf5c1b643a\files\olive-review-research\architecture`.

This persistent workspace is the designated location for later project research, historical PR analysis, Skill development, and evaluation **when separately authorized**. No historical analysis or Skill implementation was started by this consolidation. No Git repository was initialized, files published, or Olive production code/official documentation changed.
