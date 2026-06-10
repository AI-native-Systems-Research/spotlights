# loader.loadDefaultConfig.defaultScorerWeights

[← pkg_epp.config](../pkg_epp.config.md)

- **File:** [`pkg/epp/config/loader/defaults.go`](pkg/epp/config/loader/defaults.go) (lines 47–49)
- **Symbol:** `loader.loadDefaultConfig.defaultScorerWeights`
- **Kind:** config_block
- **Estimated impact:** high
- **State:** AGENT_PROPOSALS_CREATED
- **Id:** `cand-0001`

## Description
Default weights for the three scorers in the shipped no-config scheduling profile: queue depth 2.0, KV cache utilization 2.0, and prefix cache 3.0.

## Current approach
loadDefaultConfig declares three local float64 literals and wires their addresses into the default SchedulingProfile entries for queuedepth.QueueScorerType, kvcacheutilization.KvCacheUtilizationScorerType, and prefix.PrefixCacheScorerPluginType. buildSchedulerConfig in pkg/epp/config/loader/configloader.go wraps each fwksched.Scorer with scheduling.NewWeightedScorer using the configured Weight, and pkg/epp/scheduling/scheduler_profile.go aggregates endpoint scores as score*weight before the picker selects endpoints.

## Estimated impact explanation
These weights govern endpoint selection for deployments that start without an explicit config, and they sit directly on the TTFT/TPOT tradeoff: prefix affinity improves prefill reuse, while load-pressure scorers steer around queues and KV pressure. Shifting the ratio can materially move median TTFT and median TPOT on realistic multi-turn traces.

## Evolve rationale
This three-value blend is the shipped no-config routing heuristic. For multi-turn agentic traffic, increasing prefix-cache weight favors endpoints with reusable conversation prefix state and can reduce cache-hit TTFT, while increasing queue-depth or KV-utilization weight favors less saturated endpoints and can reduce TPOT under load. An evolutionary loop can mutate these positive weights while preserving the existing scorer set and non-nil Weight pointers. Correctness oracle: pkg/epp/config/loader/configloader_test.go validates the default raw profile and weights, pkg/epp/config/loader/configloader_test.go covers propagation into scheduler construction, and the scorer tests under pkg/epp/framework/plugins/scheduling/scorer/{queuedepth,kvcacheutilization,prefix} pin each scorer's score contract independently of the blend.

## Deep research proposals

### 1. Rebalance default scorer weights to trade TTFT against TPOT per Dynamo's documented tradeoff
- **Finding:** `find-0002` — *Configuration and Tuning | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.dynamo.nvidia.com/dynamo/latest/components/router/configuration-and-tuning>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In pkg/epp/config/loader/defaults.go:47-49, adjust the three default scorer weights wired into loadDefaultConfig's SchedulingProfile so the prefix-cache contribution is no longer the dominant term. Concretely, lower the prefix-cache weight from 3.0 toward parity with (or below) the load-pressure scorers, and/or raise queuedepth.QueueScorerType and kvcacheutilization.KvCacheUtilizationScorerType above 2.0, while keeping all three weights strictly positive and the existing scorer set/non-nil *float64 wiring intact so buildSchedulerConfig in pkg/epp/config/loader/configloader.go and the score*weight aggregation in pkg/epp/scheduling/scheduler_profile.go continue to behave as before. Treat the (queue, kv, prefix) triple as a single tunable point on the TTFT/TPOT frontier and search nearby ratios (e.g. (2.0, 2.0, 2.0), (2.5, 2.5, 2.0), (3.0, 3.0, 2.0)) rather than mutating any one weight in isolation. Validate via pkg/epp/config/loader/configloader_test.go for default-config and scheduler-propagation invariants and the per-scorer tests under pkg/epp/framework/plugins/scheduling/scorer/{queuedepth,kvcacheutilization,prefix}, which pin each scorer's contract independently of the blend.

**Proposal rationale.**

The finding documents a concrete, directional tradeoff from a sibling router (Dynamo): biasing routing toward prefix/KV overlap improves TTFT at the cost of inter-token latency. The candidate's shipped defaults place prefix-cache at 3.0 versus 2.0 for both load-pressure scorers, i.e. already tilted toward the TTFT side of that frontier. The caller objective is to reduce both median TTFT and median TPOT on a multi-turn agentic workload, where prefix reuse helps TTFT but queue depth and KV pressure dominate per-token latency once a session is hot. The finding therefore supplies a transferable hypothesis specific to this candidate: the current ratio is over-weighted toward TTFT for a dual-objective workload, and shifting weight from prefix toward queuedepth/kvcacheutilization should recover TPOT with limited TTFT regression. This is exactly the mutation the evolve_rationale invites, grounded in an external operator-tuning reference rather than a guess.

---

### 2. Bias default scorer weights toward prefix-cache affinity for multi-turn agentic traffic
- **Finding:** `find-0003` — *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows*
- **Source URL:** <https://arxiv.org/abs/2507.07400>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

Modify the three default float64 weight literals at pkg/epp/config/loader/defaults.go:47-49 inside loader.loadDefaultConfig so that the prefix cache scorer's weight is meaningfully larger relative to queue depth and KV cache utilization than today's 3.0/2.0/2.0 ratio (e.g., raise prefix-cache to ~4.0-5.0 while keeping queue-depth and KV-utilization at 2.0, or perform an evolutionary sweep over the three positive weights with the prefix slot as the dominant axis). The change is purely a value mutation: the scorer set, the &-of-local-literal pattern wired into the default SchedulingProfile entries for queuedepth.QueueScorerType, kvcacheutilization.KvCacheUtilizationScorerType, and prefix.PrefixCacheScorerPluginType, and the downstream NewWeightedScorer wrapping in buildSchedulerConfig (pkg/epp/config/loader/configloader.go) all stay intact, so the score*weight aggregation in pkg/epp/scheduling/scheduler_profile.go simply weights prefix-affinity more heavily when the picker chooses an endpoint. Existing oracles cover the change: pkg/epp/config/loader/configloader_test.go pins the default profile and weight propagation (update the asserted numbers), and the per-scorer tests under pkg/epp/framework/plugins/scheduling/scorer/{queuedepth,kvcacheutilization,prefix} remain untouched because individual scorer contracts are unchanged.

**Proposal rationale.**

KVFlow's central claim is that in multi-agent / multi-turn workflows, future turns are very likely to reuse earlier prefix KV state, so policies that preserve and prefer endpoints holding that prefix state reduce cache-miss TTFT. The caller context here is exactly that regime - 'multi-turn agentic workload' with a TTFT/TPOT objective - and the candidate is the only knob in the shipped no-config path that trades prefix affinity against load-pressure signals. While KVFlow itself targets eviction/prefetch inside a KV node, its workload characterization (prefix reuse dominates in agentic graphs) transfers directly to endpoint selection: the same reasoning that justifies preserving a prefix in a node justifies routing the next turn to the endpoint that already holds it. Increasing the prefix-cache scorer's weight relative to the two saturation scorers operationalizes that insight at the routing layer without requiring the deeper schema or policy work the finding also gestures at, and it is a conservative, reversible mutation bounded by the existing scorer set and tests.

---

### 3. Rebalance default scorer weights to favor live KV-cache signal over static prefix affinity
- **Finding:** `find-0008` — *KV Cache Aware Routing — production-stack*
- **Source URL:** <https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/kv-cache-aware-routing.html>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In pkg/epp/config/loader/defaults.go:47-49, mutate the three default scorer weights so that the KV-cache-utilization scorer's weight is comparable to or exceeds the prefix-cache scorer's weight, rather than the current 2.0/2.0/3.0 (queue/kvcache/prefix) ordering that puts prefix affinity highest. Concretely, candidate weight blends to evaluate include (queue=2.0, kvcache=3.0, prefix=2.0) and (queue=1.5, kvcache=3.0, prefix=2.5), keeping all three pointers non-nil and the scorer set unchanged so buildSchedulerConfig in pkg/epp/config/loader/configloader.go and the score*weight aggregation in pkg/epp/scheduling/scheduler_profile.go remain untouched. Validate with the existing default-profile assertions in pkg/epp/config/loader/configloader_test.go (updating the expected weights) and the per-scorer contracts under pkg/epp/framework/plugins/scheduling/scorer/{queuedepth,kvcacheutilization,prefix}; measure median TTFT/TPOT on a multi-turn agentic trace to confirm the shift.

**Proposal rationale.**

The finding reports that vLLM Production Stack routes to the instance with the highest KV cache hit rate and explicitly cautions against sticky prefix-only routing for multi-turn traffic where eviction invalidates affinity assumptions. The candidate's shipped default gives prefix the largest weight (3.0), which is exactly the sticky-prefix posture the finding warns about: a previously-seen prefix owner may no longer hold the KV blocks, while the KV-cache-utilization scorer reflects current cache pressure on each endpoint. Tilting the blend toward the live KV-cache signal directly addresses the gap the finding identifies and is plausible for the stated objective of reducing median TTFT and TPOT on multi-turn agentic workloads, while staying within the candidate's evolution constraints (positive weights, same scorer set, non-nil Weight pointers).

---

### 4. Rebalance default scorer weights to mirror Dynamo's prefill-credit vs decode-load cost split
- **Finding:** `find-0009` — *Routing Concepts | NVIDIA Dynamo Documentation*
- **Source URL:** <https://docs.dynamo.nvidia.com/dynamo/latest/components/router/routing-concepts>
- **Created by:** proposal_from_finding_creator

**Detailed description.**

In pkg/epp/config/loader/defaults.go:47-49, mutate the three default scorer weights so their ratio reflects Dynamo's cost decomposition: prefix-cache acts as a prefill overlap credit, while queue depth and KV-cache utilization represent decode-side load. Concretely, evolve the (queue=2.0, kv=2.0, prefix=3.0) triple toward configurations that increase the prefix weight relative to the combined load weights (e.g. prefix in the 4.0-6.0 range with queue and kv held near 2.0, or proportional reductions of the load pair), keeping all three weights strictly positive and the scorer set unchanged so existing Weight pointers and tests in pkg/epp/config/loader/configloader_test.go continue to validate. The mutation space is the relative magnitude of prefill-credit weight (prefix) vs decode-load weight (queue+kv), exposed entirely through these three float64 literals.

**Proposal rationale.**

Dynamo's cost calculation explicitly subtracts overlap credits for device/host/disk/shared cache hits from prompt-side prefill load, treating cache locality as a direct deduction against prefill work rather than a co-equal scoring axis. This maps cleanly onto the candidate's three weights: prefix cache approximates the overlap-credit term, while queue depth and KV utilization approximate the decode-side load term. For the stated multi-turn agentic workload where conversation prefixes recur, weighting prefix more heavily relative to the load pair pushes selection toward endpoints that avoid prefill recomputation, which is the dominant TTFT lever, while preserving sensitivity to decode saturation that drives TPOT. The change is a pure weight retune within the existing scorer contract, so the correctness oracles cited in the candidate (default config validation, scheduler propagation, per-scorer score contracts) remain intact.

---

## Agent proposals

### 1. Make queue-depth the dominant default weight to suppress head-of-line latency in bursty agent loops
- **Agent:** claude

**Detailed description.**

In pkg/epp/config/loader/defaults.go:47-49, mutate the three default scorer weights inside loader.loadDefaultConfig so that queuedepth.QueueScorerType becomes the strictly largest weight, with kvcacheutilization and prefix held at meaningfully smaller, near-equal values. Concrete blends to evolve toward: (queue=4.0, kv=2.0, prefix=2.0), (queue=5.0, kv=2.0, prefix=2.5), and (queue=4.0, kv=1.5, prefix=2.5), keeping all three weights strictly positive and the scorer set, addresses-of-locals wiring, and downstream NewWeightedScorer/score*weight aggregation in pkg/epp/config/loader/configloader.go and pkg/epp/scheduling/scheduler_profile.go untouched. Validate via pkg/epp/config/loader/configloader_test.go (update the asserted defaults) and the per-scorer tests under pkg/epp/framework/plugins/scheduling/scorer/{queuedepth,kvcacheutilization,prefix}, which pin each scorer's contract independently of the blend.

**Novelty rationale.**

All four listed deep_research_proposals retune within a frame where queue-depth is at most co-equal with the other scorers: find-0002 sweeps queue at 2.0-2.5, find-0003 and find-0009 push prefix to 4.0-6.0 with queue held at ~2.0, and find-0008 raises kvcache while reducing queue to 1.5. None of them make queue-depth the dominant axis. The hypothesis here is structurally different: in multi-turn agentic workloads, agent loops emit short bursts of dependent requests, so head-of-line blocking on a saturated endpoint can add seconds of TTFT that dwarfs any prefix-reuse savings, and KV utilization is a slower-moving, correlated proxy for the same saturation. Treating queue depth as the primary anti-tail-latency signal - and prefix and KV as smaller tiebreakers - is a distinct point in the weight simplex from the prefix-favored, kv-favored, and balanced rebalances already proposed.

---

### 2. Demote KV-cache utilization to a guardrail in the default blend
- **Agent:** codex

**Detailed description.**

In pkg/epp/config/loader/defaults.go:47-49, change only the default weight literals so KV cache utilization is a small positive guardrail while queue depth and prefix-cache carry the main decision, for example queueScorerWeight := 2.5, kvCacheUtilizationScorerWeight := 0.5, and prefixCacheScorerWeight := 2.5. Keep the scorer set, non-nil Weight pointers, and downstream NewWeightedScorer/score*weight aggregation unchanged. Validate by updating the default-profile assertions in pkg/epp/config/loader/configloader_test.go and measuring median TTFT/TPOT on the multi-turn agentic trace.

**Novelty rationale.**

The listed deep_research_proposals cover balanced/load-heavy retunes, prefix-dominant retunes, and KV-dominant retunes; Agent A covers queue-dominant retunes. This proposal is different: it specifically suppresses only the KV-utilization signal because the current scorer is a coarse free-KV metric that can penalize endpoints holding useful conversation KV state, while preserving queue depth as the live backlog guardrail and prefix match as the locality signal. None of the existing proposals isolate KV as the signal to demote while keeping queue and prefix co-dominant.

---
