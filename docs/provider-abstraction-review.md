# Provider abstraction review & consolidation plan

Scope: the `Provider` enum (`vc-core/src/provider.rs`) and every site that
parses it from a string, maps it to a backend/EP, tests its capabilities, or
lists the providers a build supports. This is a review of the *current*
state and a plan to aggregate the scattered logic behind `vc-core`.

## 1. Where provider knowledge lives today

`Provider` is the shared enum, but decisions *about* providers are duplicated
across at least ten sites in four crates. Each new variant (or alias) must be
threaded through all of them by hand.

| # | Site | Responsibility | Kind |
| --- | --- | --- | --- |
| 1 | `vc-core/provider.rs` | enum + `label()` + clap value-names/aliases + `is_tensorrt`/`is_cuda`/`is_windows_ml`/`is_windows_ml_directml` | canonical |
| 2 | `vc-core/model_rvc/tensorrt.rs::provider_uses_fixed_shape` | capability: TensorRT or CUDA use a fixed-shape profile | capability |
| 3 | `vc-core/model_rvc/sessions.rs::windows_ml_catalog_ep_for_provider` | `Provider` → `CatalogExecutionProvider` | mapping |
| 4 | `vc-core/model_rvc/sessions.rs::load_session` | the big `match provider { … }` that registers ORT EPs | dispatch |
| 5 | `vc-core/windows_ml.rs::provider_prepare_pending` | **second** `Provider` → `CatalogExecutionProvider` map | mapping |
| 6 | `vc-core/windows_ml.rs::CatalogExecutionProvider::vc_provider_name` | `CatalogExecutionProvider` → `"windowsml-*"` strings | mapping |
| 7 | `vc-cli/cli.rs::default_provider` | cfg-gated build default (returns `Provider`) | default |
| 8 | `vc-cli/windows_ml_eps.rs::WindowsMlEpProvider::into_catalog_provider` | CLI clap enum → `CatalogExecutionProvider` | mapping |
| 9 | `vc-gui/main.rs::parse_provider` + `provider_names` + `default_provider_name` + `gpu_device_selector_visible` | string→`Provider`, build list, default, GPU-UI predicate | parse/list/default/capability |
| 10 | `vc-vst3/config.rs::provider_enum` + `default_provider` + `gpu_provider`; `vc-vst3/editor.rs::PROVIDER_OPTIONS` + `gpu_device_selector_visible` | string→`Provider` (with aliases), default, GPU resolution, build list, GPU-UI predicate | parse/list/default/capability |

## 2. Findings

### 2.1 Three independent string→`Provider` parsers that disagree (bug)

There are three parsers and they do **not** cover the same set:

- **clap** (`provider.rs` `#[value(...)]`): all 11 variants, full alias set
  (`trt`, `winml-*`, `windowsml-dml`, `windowsml-tensorrt`, …).
- **VST3** (`config.rs::provider_enum`, lines 110–133): most variants + aliases,
  falls back to `Provider::Cpu` on anything unknown.
- **GUI** (`main.rs::parse_provider`, lines 960–971): **only 7 variants** —
  `cpu`, `cuda`, `tensorrt`, `windowsml`, `windowsml-cpu`, `windowsml-directml`,
  `windowsml-nvtrtx`. It is **missing `windowsml-openvino`, `windowsml-qnn`,
  `windowsml-migraphx`, `windowsml-vitisai`**, which the enum, clap, and VST3 all
  support.

Consequences:
- A GUI user cannot select the OpenVINO/QNN/MIGraphX/VitisAI catalog EPs, even
  though the engine can run them.
- A `settings.toml` carrying one of those provider strings makes
  `parse_provider` return `Err` → the GUI rejects an otherwise valid config.
- The VST3 parser silently maps typos to CPU (`_ => Provider::Cpu`), so a
  misspelled provider quietly runs on CPU instead of erroring.

The alias vocabulary is also maintained twice (clap `value(alias=…)` vs. the
VST3 `match` arms) and can drift.

### 2.2 Two `Provider` → `CatalogExecutionProvider` maps that must stay in sync

`sessions.rs::windows_ml_catalog_ep_for_provider` (#3) and
`windows_ml.rs::provider_prepare_pending` (#5) encode the *same* mapping
(`WindowsMlNvTensorRtRtx → NvTensorRtRtx`, `WindowsMlQnn → Qnn`, …) in two
places. A new catalog provider must be added to both or `provider_prepare_pending`
silently returns `false` (front-ends then skip the "this will download an EP"
status while `load_session` still blocks on `EnsureReady`).

`CatalogExecutionProvider::vc_provider_name` (#6) is a *third* copy of the same
correspondence, expressed as strings that also duplicate `Provider::label`
(`"windowsml-nvtrtx"`, …). These three should be one bijection.

### 2.3 Capability predicates duplicated and string-typed

- `provider_uses_fixed_shape` (#2) = `is_tensorrt() || is_cuda()`. This "is a
  GPU backend that needs a profile" concept is the same predicate the front-ends
  need for the GPU device selector — but the front-ends re-derive it from
  **strings**: `gpu_device_selector_visible` in both GUI (`main.rs:727`) and
  VST3 (`editor.rs:435`) is the identical `matches!(provider, "cuda"|"tensorrt")`.
  A future GPU provider (or the `windowsml-nvtrtx` RTX path, which also consumes
  a fixed-shape profile) will be missed by all three unless each is edited.
- The GUI/VST3 predicates operate on the raw config string, so they never see
  the canonical enum and cannot reuse `provider.rs` logic.

### 2.4 Build-availability + default expressed three different ways

"Which providers does *this* build support, and what is the default" is encoded
as:
- `vc-cli/cli.rs::default_provider` — three cfg-gated fns returning `Provider`.
- `vc-gui/main.rs::default_provider_name` + `provider_names()` — cfg-gated
  `&str` / list.
- `vc-vst3/config.rs::default_provider` + `editor.rs::PROVIDER_OPTIONS` —
  cfg-gated `&str` / `&[&str]`.

The three feature-gate ladders (`windowsml` / `tensorrt` / `cuda` / cpu-only)
are copy-pasted with slightly different arms (e.g. VST3 lists
`windowsml-directml` in its options, the GUI list differs, the CLI leans on
clap). There is no single "providers available in this build" source, so the
lists can — and already do (§2.1) — diverge.

### 2.5 `load_session` bail messages repeat the cfg story per arm

Every `Provider::WindowsMl*` arm in `load_session` repeats the same
`#[cfg(not(all(windows, feature = "windowsml")))] bail!("… rebuild on Windows
with the windowsml feature …")` block (four near-identical copies). A single
"is this provider compiled into this build" gate would remove the repetition.

### 2.6 What's already good (keep)

- `Provider` being `Copy` + backend-neutral, with the clap derive gated behind a
  feature so plugins don't pull clap — good separation.
- `CATALOG_PRIORITY` / `CatalogExecutionProvider::from_catalog_name` already
  centralize the *catalog-name* side; the gap is only the `Provider` side.
- The session structs (`HubertEmbedderSession` / `RmvpePitchSession` /
  `RvcModelSession`) already funnel every backend through one `extract`/`infer`
  contract — that abstraction is sound and is **not** what this plan touches.

## 3. Consolidation plan

Goal: make `Provider` the single source of truth so adding a variant means
editing one table, and eliminate the divergences in §2. Order is
lowest-risk-first; each step compiles and ships on its own.

> **Status:** Steps 1–4 done and verified (cpu + windowsml builds; tensorrt/cuda
> builds unverified — no GPU SDK on the dev box). Step 5's tests landed alongside
> them. Deferred by design (see notes below): the curated front-end **option
> lists** (`provider_names` / `PROVIDER_OPTIONS`), the repeated `load_session`
> windowsml bail arms, and the CLI's `WindowsMlEpProvider` arg enum.
>
> Step 4 decision (agreed): a single-provider build defaults to its backend and
> the **cuda-only** build now defaults to `cuda` (was `cpu` for the CLI/GUI;
> cuda is dev-only/unpackaged, so the change is low-impact). The combined dev
> build (windowsml + tensorrt) and the CPU-only build still default to `cpu`.

### Step 1 — One parser: `impl FromStr for Provider` (+ `all()` / `aliases()`)

- Add a provider descriptor table in `provider.rs` (one row per variant:
  canonical name, aliases, label). Drive `label()`, the clap value-names, and a
  new `FromStr` from that table so the canonical/alias vocabulary is defined
  once.
- Replace `vc-gui::parse_provider` and `vc-vst3::provider_enum` with
  `Provider::from_str`. This alone fixes §2.1 (GUI gains the four missing
  variants; VST3 stops silently swallowing typos — decide explicitly whether an
  unknown string is an error or falls back to CPU, and do it in one place).
- Keep clap working by pointing its parsing at the same table (or keep the
  derive but assert in a test that clap's accepted set == `FromStr`'s set).

### Step 2 — One capability surface on `Provider`

- Move `provider_uses_fixed_shape` onto the enum as
  `Provider::uses_fixed_shape(self)` (keep the free fn as a thin re-export for
  `tensorrt.rs`, or update call sites).
- Add `Provider::shows_gpu_device_selector(self)` (initially
  `uses_fixed_shape()`), and have both front-ends call it after parsing the
  string to a `Provider`, deleting the two `matches!(…, "cuda"|"tensorrt")`
  copies in §2.3. This also auto-corrects the `windowsml-nvtrtx` case if we
  later decide it should show the GPU selector.

### Step 3 — One `Provider` ↔ catalog EP bijection

- Add `Provider::catalog_ep(self) -> Option<CatalogExecutionProvider>` (the
  single mapping) and `CatalogExecutionProvider::vc_provider(self) -> Provider`
  for the inverse.
- Rewrite `windows_ml_catalog_ep_for_provider` (#3) and
  `provider_prepare_pending` (#5) to call `catalog_ep()`; derive
  `vc_provider_name` (#6) from `vc_provider().label()`. This collapses the three
  copies in §2.2 into one, so a new catalog EP is a single table entry.
- Fold the CLI `WindowsMlEpProvider::into_catalog_provider` (#8) into the same
  mapping where practical (the CLI clap enum can map through `Provider`).

### Step 4 — One build-availability source

- Add `Provider::available_in_build(self) -> bool` (cfg-driven, centralizing the
  §2.5 gate) and `providers_for_build() -> &'static [Provider]` /
  `default_provider() -> Provider` in `vc-core`, cfg-gated once.
- Replace the three default ladders (§2.4): CLI `default_provider`, GUI
  `default_provider_name`/`provider_names`, VST3 `default_provider`/
  `PROVIDER_OPTIONS` become thin wrappers that render `providers_for_build()`
  (front-ends only need `label()` for display). GPU-resolution helpers
  (`vst3::gpu_provider`) can then validate against `available_in_build()`.
- Simplify the repeated `load_session` bail arms (§2.5) using
  `available_in_build()`.

### Step 5 — Lock it with tests

- Round-trip test: for every `Provider`, `from_str(label()) == provider`, and
  every alias parses.
- Parity test: clap's accepted value set == `FromStr`'s accepted set.
- Bijection test: `catalog_ep().map(vc_provider) == Some(self)` for the
  windowsml catalog variants; `from_catalog_name`/`vc_provider_name` agree.
- A doc/test note that "adding a `Provider` variant = add one table row + wire
  `available_in_build`/`catalog_ep`", so future variants can't reintroduce the
  drift.

### Risk / sequencing notes

- Steps 1–3 are `vc-core`-internal + mechanical front-end swaps; low risk.
- Step 4 touches the cfg ladders that gate shipped builds — verify each package
  variant (`windowsml`, `tensorrt`, cpu-only, `cuda` dev) still resolves the
  same default and lists the same providers it does today (except the GUI, which
  *should* now gain the four catalog EPs — confirm that's intended before
  shipping).
- None of this touches the realtime worker, chunk grid, or the
  session/`extract`/`infer` contract, so it carries no audio-quality risk.
