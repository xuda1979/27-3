# Qwen 27B to 3B Quantum Coding Research and Development Plan

Structural compression and verified recovery training

16 November 2026  |  Version 1.0  |  Research proposal


### Recommended decision

Develop a genuine model with at most 3 billion deployed parameters by combining structured width and depth reduction, selective low rank factorization, and teacher guided healing. Train quantum coding competence throughout recovery, while retaining general code, mathematics and instruction following. Quantize only after a viable small model exists, then recover the quantized model against the original teacher.

Run a second, independent route that distills Qwen3.8-27B into an existing small model. This is the practical baseline and fallback. It transfers capability without claiming that the small model was produced by pruning the 27B weights. Continue structural compression only if its quality, latency or total cost justifies the additional engineering.

| Route | Starting point | Purpose |
| --- | --- | --- |
| A  Structural compression | Qwen3.8-27B → intermediate models → about 2.95B | Main research route; inherited weights and custom architecture |
| B  Small model distillation | Official Qwen3.5-2B plus the same teacher and verified data | Early useful model and a strong cost comparison |
| C  Quantized teacher | Unchanged 27B architecture at supported low precision | Deployment control; remains a 27B model |


### What success means

The deliverable is numerical weights that load independently, pass fresh quantum programming tests, and reduce measured inference cost. A smaller file, lower training loss or a perfect score on a small public benchmark is insufficient. The principal comparison is verified task success at a fixed inference budget; report parameters, bytes, latency and training cost separately.

The 9-fold parameter reduction is an ambitious research target. Published compression results do not establish that Qwen3.8-27B can retain its full reasoning ability at 3B. The first investment buys evidence about recoverability, not a promise of full recovery. The proposed thresholds, data volumes and hyperparameters below are starting decisions, not measured results.


### Relationship to quantum computing gpt

The current repository plan specifies a 27B RL-only adapter and explicitly excludes compression and teacher imitation. This new request defines a separate compression study with supervised distillation and healing. Reuse its data isolation and verification principles, but create a new experiment contract rather than relabeling the existing SAPO route. The reviewed repository describes production SAPO and full evaluation services as work still to implement. [S19]


## 1  Model definition and parameter budget

Use the official Qwen/Qwen3.8-27B release, pinned by immutable revision and weight hashes. Its text model has 64 layers with a repeating three Gated DeltaNet plus one full-attention pattern. The configuration has hidden width 5120, FFN width 17408 and an untied vocabulary of 248320 entries. The architecture is dense, not a 3B-active mixture of experts. [S01]

Define the target as at most 3,000,000,000 distinct deployed scalar parameters, including embeddings, output head, retained adapters and correction modules. Store tied tensors once, but also report logical module sizes. Report quantization scales and other buffers in deployed bytes. Remove vision and unused MTP modules from the text-only export; freezing them does not remove their storage.


### Initial dense candidate

| Field | Candidate setting |
| --- | --- |
| Layer layout | 32 layers: 24 Gated DeltaNet and 8 full-attention layers |
| Hidden and FFN width | 2048 and 6144 |
| Full attention | 12 query heads, 2 KV heads, head dimension 256; gated query projection |
| Gated DeltaNet | 8 key/query heads, 24 value heads; head dimensions 128; convolution width 4 |
| Tokenizer and output | Keep all 248320 token IDs; untied input embedding and output head |
| Initial context target | 8K validated first, then 16K and 32K after recovery |

| Component | Calculated parameters |
| --- | --- |
| Input embedding and output head | 1,017,118,720 |
| FFN matrices | 1,207,959,552 |
| Gated DeltaNet modules | 556,503,168 |
| Full-attention modules | 167,776,256 |
| Residual and final normalization | 133,120 |
| Total excluding vision and MTP | 2,949,490,816 |

This count is analytical, independently checked against the inspected module shapes, and must be confirmed by constructing the pinned model. It leaves 50.51M parameters below the cap. Retained adapters consume that margin. The attention code permits hidden width to differ from query-head count times head dimension. [S02]

A key constraint is vocabulary cost: the original two 248320 × 5120 matrices contain 2.543B parameters. Width reduction is therefore essential. The candidate retains the vocabulary to preserve arbitrary code, Unicode and existing token semantics; vocabulary trimming is a separate, higher-risk experiment.


## 2  Evidence and method selection

The strongest plan combines established components with a few isolated research hypotheses. Published results below motivate experiments; none is a measured result for this project.

| Method | Evidence and useful mechanism | Decision |
| --- | --- | --- |
| Minitron 2024 | Structured pruning followed by distillation; the 15B-to-4B example used 94B recovery tokens. [S03] | Use as the main reproducible recipe. |
| Minitron SSM 2025 | Group-aware compression of hybrid models. Its 8B-to-4B study used about 380B final KD tokens. [S04] | Adapt dependency groups to DeltaNet; do not reuse Mamba surgery unchanged. |
| SVD LLM V2 2025 | Allocates compression across matrices and improves singular-value truncation. [S05] | Low rank baseline after activation calibration. |
| DELTA 2026 | Joint rank selection and asymmetric quantization of the two factors under a memory budget. [S06] | Research arm with additional parameter and latency constraints. |
| On-policy distillation | MiniLLM, DistiLLM-2 and GKD address student/teacher distribution mismatch in different ways. [S07–S09] | Use a simple forward-KL baseline, then test student-prefix training. |
| Quantization Aware Healing 2026 | Recovers a structurally compressed quantized student using the original teacher. [S10] | Compare with matched BF16 continued KD and PTQ controls. |
| AWQ and SpinQuant | Activation-aware scaling and learned rotations reduce quantization error. [S11–S12] | Start with supported weight-only quantization; rotations need hybrid parity tests. |


### What the evidence does not justify

Quantization changes precision, not parameter count. QLoRA reduces adaptation memory by training small updates through a frozen quantized backbone; it does not replace that backbone with a 3B architecture. Low rank compression replaces a dense matrix with smaller factors, whereas LoRA normally adds factors to the existing matrix. [S13]

Unstructured zeros need sparse kernels to produce useful deployment savings. A model with 3B active experts can still store many more parameters. Tensor-train or tensor-network decompositions are optional alternatives only after a measured kernel prototype shows an advantage over dense structured pruning and two-factor SVD.

Novelty should come from a validated quantum-specific allocation or recovery mechanism. Combining known compression methods alone does not establish an original algorithm. Treat all proposed innovations in Sections 5 and 6 as hypotheses pending ablations and a dedicated novelty review.


## 3  Structural compression procedure


### Resolve the native model before cutting weights

Pin model, tokenizer, processor, chat template and a compatible implementation. Compare a trusted unmodified reference loader with the proposed training loader on fixed prompts: logits, greedy outputs, prefill, cached decoding and long-prefix continuation. Use measured numerical tolerances from the unmodified reference. The fetched config specifies output_gate_type="swish", while the inspected moving Transformers implementation applies a sigmoid attention gate. This is an unresolved semantic compatibility signal, not proof that either source is wrong. Resolve it before any pruning. [S01–S02]


### Search a small architecture set

Start with the 2.95B candidate, a deeper/narrower candidate, and a shallower/wider candidate, all respecting the same parameter cap. Compare direct pruning against a staged schedule around 27B → 14B → 7B → 2.95B. These are candidate waypoints, not mandatory architectures. If one cut causes collapse, split it into smaller cuts; account for every intermediate recovery token. Compare schedules at equal total compute as well as equal final-model exposure.

Use a mixed training calibration subset of 1024 sequences at 2K–8K tokens, balanced across quantum code, general Python, mathematics and tool use. Obtain cheap activation and reconstruction scores, shortlist no more than six paths, and use small recovery runs to rank them. Zero-shot pruning loss alone is unreliable after large surgery.


### Preserve connected parameter groups

Residual width: select a consistent residual-channel basis across embeddings, normalization and every input/output projection. Recompute shapes and initialization scales. Width surgery changes RMS normalization statistics; fit affected linear projections against teacher activations before full healing. Arbitrary rotation through nonlinear or gated modules is not an exact invariance.

FFNs: prune corresponding gate/up rows and down columns together. Full attention: select a KV head with its associated query-head group and gate channels; update output columns, cache layout and RoPE metadata. DeltaNet: jointly select Q/K groups, their value heads, convolution channels, a/b gates, time constants, normalization and output columns. Preserve whole head dimensions initially.

Depth: retain the three-to-one mixer pattern in the first implementation, compare complete cycles, and preserve coverage across early, middle and late positions. Do not assume every early or late layer is indispensable; measure it. Reindex layers and caches explicitly. Later irregular layouts require their own kernel and learning tests.


### Mandatory surgery tests

Test an identity configuration that preserves the original model; verify no accidental changes when retaining all dimensions. Test every compressed shape, parameter count, save/load, optimizer update, causal masking and prefill/decode agreement. Check recurrent resets between packed examples. A valid tensor shape is not evidence of preserved behavior.


## 4  Low rank compression and reconstruction

Apply low rank approximation selectively to FFN and large projection matrices that retain redundant directions after structural reduction. Avoid compressing small gates, normalization, decay parameters and embeddings in the first arm. Protect sensitive modules based on development evidence rather than a fixed rule that all attention is important and all FFNs are disposable.

W: m × n   →   A: m × r, B: r × n;   W_hat = A B

P_original = m n;   P_factorized = r (m + n)

Compression occurs only when r < mn/(m+n). For a 2048 × 6144 matrix, rank 768 uses 6.29M parameters instead of 12.58M. This arithmetic does not imply equal accuracy or faster execution; two smaller matrix multiplications can be slower than one well-optimized dense operation.


### Fit the layer outputs seen in real tasks

min over A,B  E_x [ || W x - A B x ||² ]

Estimate input covariance from the mixed calibration set. Use regularized whitening or an equivalent numerically stable activation-aware factorization; choose shrinkage and numerical thresholds on training calibration data. Refit downstream layers using compressed upstream activations to reduce accumulated mismatch. Maintain a uniform-rank SVD control and the stronger SVD-LLM V2 baseline. [S05]

Allocate ranks per module, not uniformly. Enforce three independent limits: sum of actual factor parameters ≤ 3B; packed deployment bytes below the chosen memory target; and measured latency on the target runtime. Candidate ranks and widths should align with supported kernel tiles. Do not promise acceleration from FLOP counts alone.


### Factor precision is a separate variable

DELTA motivates allowing different precisions for A and B, such as 4 and 8 bits, while searching retained rank. Its original memory objective can keep more low-bit parameters. Our search must also enforce the 3B scalar-parameter ceiling. Compare symmetric quantization at the same parameter count and bytes, and report the residual tradeoff when both cannot be matched exactly. [S06]


### Keep the exported model small

Store the factorized modules as factors. Materializing A B at export restores the dense matrix and may erase the reduction. Temporary training projections can be discarded only if inference never calls them. A trainable LoRA residual on a factor must either be merged into that factor or included in the parameter count; merging it into the original full-size matrix is not a compressed release.

Use full-student healing as the primary recovery path. Low rank adapters are a cheaper ablation, but constraining all recovery updates to tiny adapters after destructive 9-fold compression may prevent repair. Test the tradeoff instead of assuming parameter-efficient fine-tuning is sufficient.


## 5  Proposed innovation in compression allocation


### Hypothesis A  Allocate capacity by semantic failure risk

Average reconstruction loss may undervalue rare but decisive quantum operations: controlled phase, inverse circuits, qubit ordering, parameter binding and measurement semantics. A small numerical distortion can preserve fluent code while destroying these capabilities. The proposed allocation scores candidate cuts on both activation reconstruction and executable quantum task damage.

score(cut) = reconstruction_error + a × mean_failure_increase

             + b × worst_family_failure_increase

All scores come from training-only calibration roots. Use sampled ablations to estimate which groups are sensitive, then a constrained greedy or dynamic-programming allocator over hardware-valid cuts. Evaluate semantic risk only for shortlisted configurations to avoid running simulators for every tensor. Re-estimate after major cuts because importance is not additive.

Protect minimum capacity for all declared skill families, including mathematics and general coding. Use held-out development tasks to select weights a and b from a small predeclared grid; do not access final tests. Confirm that higher semantic scores predict unseen family failures better than reconstruction alone.


### Experiment and falsification

Compare three initializations at equal actual parameters: uniform cuts, activation-aware allocation, and activation plus semantic-risk allocation. Hold healing data, tokens, optimizer, teacher and export precision fixed. The proposed method earns adoption only if fresh-task pass@1 improves with a positive paired confidence interval, without violating latency or general-code guardrails. If gains disappear after ordinary healing, the allocation is not worth its extra calibration cost.


### Hypothesis B  Allocate rank and precision jointly

After the dense or low rank candidate works, search factor ranks and 4/8/16-bit precision together with the semantic-risk term. Use a worst-family constraint so common circuit templates cannot consume all capacity. Include parameter count, file bytes and target-runtime latency in the search contract. Retain BF16 in recurrent decay and normalization until a separate stability study supports changing them.

The proposed contribution is the semantic, family-constrained allocation and its measured effect on quantum code; asymmetric factor quantization itself already has a published precedent. Compare against DELTA-style reconstruction-only allocation using the same candidate set and total search budget. [S06]


### Expected result and stop rule

Expected benefit is better preservation of rare correctness-critical skills, not recovery of arbitrary lost knowledge. Stop this branch if calibration decisions are unstable across two resamples, if the advantage is underpowered after the planned confirmation, or if extra kernels make verified-task latency worse. Record negative findings; do not keep adding objective terms until a test score improves.


## 6  Proposed innovation in healing


### Hypothesis C  Preserve useful memory through observable probes

DeltaNet recurrent states change size after head and width pruning. Direct mean-squared error between teacher and student state tensors is not meaningful without a justified alignment. Instead, give both models the same prefix and ask continuation questions that require facts from that prefix: register mapping, an earlier circuit definition, a parameter convention or a tool result.

L_memory = mean KL( p_teacher(continuation | prefix),

                    p_student(continuation | prefix) )

Create counterfactual prefix pairs differing in one relevant fact. The continuation must change accordingly. Verify the resulting program, not merely whether it quotes the prefix. Compare ordinary next-token distillation, ordinary longer-context data, and these targeted probes with the same sequence lengths, tokens and teacher calls.

Backpropagate through the prefix during the training experiment, using activation checkpointing where supported. A detached cached prefix can diagnose memory but cannot repair all earlier memory formation. Use clean recurrent resets at each document boundary. Begin with 2K/8K prefixes, then test 16K/32K only after throughput and recovery gates pass.


### Hypothesis D  Repair the errors the student actually makes

Generate candidates with the current student on training tasks. Execute them in an isolated evaluator. Classify failures into API misuse, syntax, qubit ordering, phase/control semantics, numerical assumptions, incomplete tests and tool workflow errors. Ask the teacher for a concise repair conditioned on the same task, failed code and public execution feedback; accept a repair only after independent tests pass.

Train on verified repairs and on teacher distributions at student-generated prefixes. Keep successful alternative student solutions as positive data; never label them wrong simply because their text differs from the teacher. Cap repeats per root, retain failed attempts in the audit ledger and stratify the repair queue so easy API fixes cannot dominate.

Compare offline teacher-only training, ordinary on-policy distillation and verified error-specific repair at equal teacher generation plus scoring cost. GKD and MiniLLM supply the distribution-mismatch motivation; the domain contribution is verification, error taxonomy and controlled allocation, not the invention of on-policy distillation. [S07–S09]


### Teacher selection

The untouched 27B model supplies broad retention targets. A separately validated quantum specialist may supply verified code and repairs if it demonstrably improves on that teacher. Record the teacher for every sample and evaluate it on disjoint development data. Never assume the existing repository already contains a strong trained adapter. A failed teacher solution cannot become a trusted hard target.


## 7  Data factory and mixture

Build the evaluator and data ancestry system before large-scale generation. Separate broad recovery tokens from expensive executable quantum episodes. Repeated variants of a few circuit templates cannot replace a broad recovery corpus.

| Token domain | Early recovery | Quantum specialization |
| --- | --- | --- |
| Quantum code and APIs | 25% | 45% |
| General Python and software | 30% | 20% |
| Quantum mathematics and physics | 20% | 15% |
| Tool use and research workflows | 10% | 15% |
| General instructions and language | 15% | 5% |

Shares refer to effective training tokens, not raw record counts, and sum to 100% in each column. Within quantum code begin with 60% Qiskit, 20% Cirq and 20% PennyLane tokens. Keep independent skill labels for algorithms, state preparation, decomposition, variational methods, noise/channels, compilation, debugging and API migration. These are proposed mixtures; adapt them only from development evidence.


### Initial corpus construction

Pilot target: 5,000 distinct quantum training roots, 500 development roots and 1,200 sealed fresh test roots. Generate 2–8 variants per training root, targeting 20K–40K independently verified episodes. At a mean 4K tokens this provides roughly 80M–160M episode tokens before filtering, not billions of unique quantum examples. Add versioned documentation, licensed code and broad code/math text for larger recovery runs.

Scale toward 50K–200K roots only after the pilot exposes genuine coverage gaps. Count accepted unique roots, accepted variants, repeated exposures and effective tokens separately. Report source licenses and redistribution constraints in manifests. Do not fill a 100B-token budget by recycling a tiny synthetic pool.


### Task record and acceptance

Each root stores source identity and revision, ancestry/family ID, SDK lock, prompt, public interface, independently created private tests, a reference solution, numerical tolerances, difficulty and resource limits. Every episode additionally stores model revision, sampling settings, output, execution trace, verdict and failure category. References and private tests remain outside actor-visible context.

Admit a task only when the reference passes, at least one distinct correct implementation passes, and relevant incorrect mutants fail. Split by source/algorithm ancestry before variants, translations or repairs are produced. Keep calibration roots inside training. Use exact, approximate text, AST and semantic-family checks to detect leakage; public benchmark contamination inherited from pretraining cannot be ruled out by project deduplication.


## 8  Quantum verification and difficult examples

Pin separate SDK environments for every evaluation contract. A benchmark written against an old Qiskit API should run under its native version; a current-API migration task is a separate measurement. A successful import or matching output string is not a semantic quantum test.


### Verification ladder

First check parsing, required function signatures, inputs and return types. Then execute in a sandbox with no network, no reference files and bounded CPU, memory and wall time. Require evidence that the candidate function and every intended assertion ran. Include timeout and invalid-output cases in the denominator; infrastructure failures trigger an invalid measurement and a controlled rerun, not a score increase.

For small circuits, use exact statevector or operator comparison with declared conventions. State equivalence uses fidelity |<psi_ref|psi_candidate>|² and a fixed numerical tolerance, initially 1 − fidelity ≤ 1e−8 for deterministic small double-precision checks. Compare unitaries up to global phase only when the task contract permits it. Do not discard a global phase before turning an operation into a controlled operation: it can become observable relative phase.

For channels, compare density matrices or Choi representations at small sizes, including trace preservation and positivity when required. For sampled measurements, use prespecified confidence bounds and shot budgets; finite-shot frequencies are not expected to match exactly. Use stabilizer, tensor-network or algebraic checks where applicable; keep exact simulation mostly at 2–8 qubits in the pilot and size-test before extending beyond that.

| Failure family | Concrete task and hidden check |
| --- | --- |
| Qubit order | Prepare a basis state on a permuted register; test several nontrivial mappings, not only all-zero input. |
| Controlled phase | Implement a controlled phase or inverse QFT; compare operator action on superpositions. |
| Parameter binding | Build a variational circuit; test unseen angles and a finite-difference gradient. |
| Quantum channels | Implement amplitude damping; test multiple input states and trace preservation. |
| Generalization | Translate an algorithm between SDKs or repair a changed API while preserving semantics. |


### Avoid rewarding the verifier rather than the task

Use independent reference implementations and metamorphic checks: U followed by U inverse, known symmetries, register permutations and conservation laws when valid. Reject no-op circuits, hardcoded outputs and test-file access. Mutate control/target order, rotation sign, measurement order and return types to verify that the evaluator catches plausible errors. A 2026 quantum-benchmark audit documents why evaluator validity itself needs testing. [S18]


## 9  Recovery objectives and implementation details

Train the newly compressed student in BF16 first. Keep the teacher frozen. The default recovery objective combines correct hard targets and teacher distributions, with auxiliary representation matching only in a separately measured arm.

L = 0.3 L_CE + 0.7 T² mean KL(p_teacher,T || p_student,T)

T = 1 initially; test T = 2 in a bounded ablation

CE uses trusted corpus next tokens or independently verified response tokens. For conversational examples mask user, system and tool-result tokens from CE; for raw corpus recovery use the declared causal-language-model mask. Define KD masks explicitly and match teacher/student prefixes. Normalize each loss by its valid token count before combining; log unweighted losses and weighted contributions. When the teacher answer fails quantum verification, set response KD weight to zero and use the verified CE target; renormalize active weights and record the exclusion.


### Token and template alignment

Retaining the original tokenizer enables token-level distillation only after token-to-ID maps, normalization, added tokens and special-token semantics match exactly. Equal vocabulary size is insufficient. For Qwen3.5-2B verify these identities explicitly. For a different tokenizer use verified text/sequence targets or a separately tested alignment method; do not compare same-index logits. Teacher and student may need distinct native templates, which must preserve the same semantic conversation.


### Avoid full vocabulary tensors in memory

At 8K tokens and 248320 vocabulary entries, one BF16 dense logit tensor for one sequence is about 4.07 GB. Stream sequence chunks and use memory-efficient KL computation. Full-logit caching for 1B tokens would require about 497 TB. Prefer online chunked scoring or selective caching. Top-64 targets with 32-bit IDs and 16-bit values already cost roughly 384 GB per billion tokens before metadata.

If using top-k logits, preserve the exact teacher log-normalizer and the tail probability. Compare k = 32/64/128 on a small exact-KL fixture. An aggregated-tail KL is an approximation, not full KL; document the approximation and verify numerical gradients. Use the same approximation across experimental arms.


### Optional representation calibration

Match selected hidden-layer outputs with temporary learned projections, initialized by least squares when feasible. Map student layers monotonically to teacher layers and normalize the loss to avoid scale domination. Use this only if it improves recovery over logit KD at matched compute. Remove temporary projections from the inference graph and parameter report only after a fresh export proves that they are unused.


## 10  Training stages and starting settings

| Stage | Work | Exit evidence |
| --- | --- | --- |
| 0  Native preflight | Resolve model/runtime; baseline inference; real forward/backward and save/resume. | Native parity, correct gradients, clean reload and authenticated baseline. |
| 1  Surgery screening | Up to six paths, 50M student tokens per path; top three receive 200M additional each. | 0.9B aggregate screening tokens; numerical health and improving development scores. |
| 2  Recovery finalists | Two paths receive up to 1B additional student tokens each. | Learning curves plus quality/cost comparison; do not rank overlapping candidates prematurely. |
| 3  Quantum healing | Use accepted code/repair data; switch mixture gradually; bounded on-policy study. | Fresh development pass@1 improvement with retention guardrails. |
| 4  Verification RL | Optional only after reward diversity and a successful lifecycle test. | Benefit over equal-cost continued distillation. |
| 5  Quantized recovery | PTQ control, quantization-aware KD, matched BF16 continued KD. | Packed export re-evaluation and measured latency. |
| 6  Confirmation | Selected method and strongest control on independent seeds; sealed tests last. | Complete denominators, intervals, cost and numerical weights. |

Stages 3–5 share a proposed 0.8B additional student-token ceiling in the first feasibility program, giving 3.7B total across all paths. Count intermediate-model healing within each path allocation. This ceiling excludes neither failed steps nor repeats: log attempted, effective and billed work. Confirmation and full-scale extension require a new measured allocation; these figures are planning budgets, not resource reservations.


### Initial hyperparameters

Use AdamW with beta1 = 0.9, beta2 = 0.95, epsilon = 1e−8 and gradient clipping at norm 1.0. Start full-student recovery at peak LR 2e−5; test 1e−5 and 5e−5 in a short canary only. Use 2% warmup and cosine decay to 10% of the peak. Begin weight decay at 0.01 for matrix weights and zero for bias/norm parameters. These are conservative proposals, not paper-mandated values.

Start with 2K sequence length, then 8K after stable updates. Target an effective batch of 131072 tokens, adjusted from measured memory and gradient noise. Pack only if both attention and recurrent states reset across documents; otherwise use independent examples. Log effective tokens rather than assuming every packed position contributes. Begin quantum specialization at LR 1e−5 and reduce it if general-code retention degrades.

If a candidate loses over 5 percentage points on the protected development retention suite after a stage, pause and compare against its pre-stage checkpoint. A failed gate leads to diagnosis, smaller cuts or a different route; it does not justify quietly relaxing the target.


## 11  On policy learning and verification based RL


### Introduce student prefixes gradually

Begin with offline verified examples and broad recovery KD. After the student produces useful code, allocate 20% of response-training tokens to student-generated prefixes, increasing to at most 50% only if held-out performance improves. Teacher scoring must use those same prefixes, not substitute its own trajectory. Keep replay data to prevent collapse toward a narrow failure set.

Use ordinary forward KL as the reference; compare GKD/Jensen-Shannon or a pinned MiniLLM-style objective as one changed factor. Do not casually replace forward KL with reverse KL and retain the same estimator: sampling distribution and gradient estimator matter. DistiLLM-2 is a separate published contrastive baseline, not a synonym for every repair loss. [S07–S09]


### Start RL only when the student can receive useful reward

Require finite gradients, a working resume path and at least 30% of sampled task groups with nonconstant binary rewards in a 256-task training canary. Sample eight candidates per task initially. A pool that produces almost all failures needs easier tasks or more supervised healing; repeated zero-variance groups do not create a learning signal.

R(program) = 1 if every required private test passes; otherwise 0

Use binary semantic success first. Syntax, compilation and partial tests are diagnostics, not independent reward bonuses in the initial arm. Keep prompts, public tools and private rewards separate. Mask tool outputs, prompts and padding from the actor loss. Normalize group advantages with a defined zero-variance policy; record all candidates and skipped groups without hiding their cost.

Run a pinned GRPO reference against the project-preferred SAPO with identical initialization, prompts, sampler, reward, reference policy and rollout budget. SAPO changes the policy-update rule; it does not repair a bad verifier or create missing successful trajectories. Use the verified implementation and loss-gradient tests from the paper/runtime contract rather than inventing a new formula. [S14]


### Bound the first experiment

Start full-student RL at LR 5e−7 to 1e−6, one policy pass per rollout batch and KL coefficient 0.01 as a canary setting. Use current-policy generations, record behavior log probabilities and reject stale rollouts. Set 50M generated-token and compute ceilings inside the Stage 3–5 allocation. Compare against continued verified KD with the same total teacher/actor/verifier cost.

Adapt task difficulty to keep estimated training success mostly between 20% and 70%, while maintaining every domain floor. Update mixture weights at declared development intervals, cap any change at five percentage points per update and retain a fixed-mixture control. Final-test failures never enter the repair queue or curriculum.


## 12  Quantization and target runtime

Keep a BF16 student as the scientific reference. First quantize supported linear modules to W8A16 or W4A16 using mixed-domain calibration. W4A16 means four-bit weights with sixteen-bit activations; it does not promise four-bit recurrent state, caches or all embeddings. Use group size 128 as an initial candidate and compare 64 only where supported.


### Recover against the original teacher

Test quantization-aware KD from the original 27B teacher, using the same correct quantum examples and broad replay. Keep high-precision master weights during fake-quant training; the exported artifact must use real packed low-bit weights. Compare four arms: BF16 checkpoint, BF16 plus continued KD, PTQ of that checkpoint, and quantized forward-pass KD with the same extra data. This isolates quantization awareness from simply adding more training. [S10]

Keep gates, decay/time parameters, normalization and recurrent state in the validated precision initially, including FP32 accumulation where the implementation requires it. Mixed precision for factor matrices is a later arm. Learned rotations must preserve the full-precision hybrid computation before low-bit training; do not push a Transformer rotation through DeltaNet gates or SiLU without proving equivalence.

| Artifact | Raw weight storage estimate |
| --- | --- |
| 27B at BF16 | 54.0 GB before vision, auxiliary modules and runtime state |
| 27B at uniform 4 bit | 13.5 GB lower bound; still 27B parameters |
| 2.94949B student at BF16 | 5.899 GB |
| Same student at uniform 4 bit | 1.475 GB theoretical lower bound |
| Student with embeddings/head BF16, remaining weights 4 bit | 3.000 GB before scales, alignment and buffers |

Report actual file bytes and peak resident memory. For the candidate, eight attention layers with two KV heads of width 256 require about 134 MB of BF16 KV cache at batch one and 8K tokens, before overhead. DeltaNet recurrent state is additional; its storage is not an ordinary sequence-length KV cache.


### Ascend deployment is a separate acceptance gate

Assume the project may use Ascend 910B/910C, but verify assigned hardware and memory. Pin CANN, torch-npu, Transformers, inference engine and training dependencies. Published Qwen-family inference or quantization examples do not certify backward kernels, custom shapes or this exact 3B export. CUDA-only bitsandbytes or Marlin kernels cannot be assumed to work on Ascend. [S15–S16]

Test the full lifecycle and packed inference on the target device. If custom low rank layers or INT4 lack fast kernels, retain dense structured BF16/INT8 as a valid deployment option. Do not choose a smaller checkpoint that increases task latency.


## 13  Evaluation and acceptance criteria

Freeze task prompts, environments, scorer, decoding, tool access and budgets before final comparison. Report native single-generation model results separately from a system allowed to retrieve, execute and repair. Use the same tool harness for teacher and student when making a system comparison.

| Evaluation | Protocol and role |
| --- | --- |
| Qiskit HumanEval Standard and Hard | Run the complete pinned public suites in their native environments; report solved/total and exact variant. Standard has 151 tasks. [S17] |
| Fresh private quantum set | Primary result: 1200 independent roots across SDKs, algorithms and failure families. Keep inaccessible during training and architecture selection. |
| General code and math | Pinned HumanEval+/MBPP+ or licensed current code tests; a fixed math/physics retention suite. Audit versions and denominators before use. |
| Interactive research | Held-out repair and investigation episodes; identical tools and wall-time/token caps. Report success, calls, tokens and elapsed time. |
| Runtime | Same device and engine where possible; batch1 and batch8, 2K/8K/32K inputs, fixed output lengths. Measure TTFT, decode speed and peak memory. |


### Preregistered proposed gates

Feasibility: the 3B candidate reaches at least 75% of the measured 27B teacher pass@1 on the development quantum suite, and its recovery curve is still improving or it is competitive with the small-model route. This is a continuation gate, not release quality. If it fails, retain useful intermediate models and diagnose; the 3B objective remains unmet.

Release target: ≤3B parameters; at least 90% of teacher quantum pass@1; and an absolute improvement of at least five percentage points over the strongest tested ≤3B small-model baseline, with a positive paired 95% confidence interval. Also require general-code and math noninferiority margins of five percentage points against the declared pre-specialization student control. Full 27B general capability parity is not assumed.

Quantized release: one-sided 95% confidence bound supports no more than two percentage points loss versus the matched BF16 student on the primary task suite. Efficiency target: at least 2× lower median end-to-end task latency versus the 27B deployment control at equal success protocol, with a reported confidence interval. A failed target is reported explicitly, not omitted.


### Uncertainty and leakage

Use paired bootstrap intervals clustered by task ancestry and paired correctness tests, with multiplicity correction for secondary claims. Confirm the selected method and control with three training seeds when resources permit; one seed is a pilot. Around 1200 independent tasks, a single 80% score has roughly ±2.3 percentage points nominal 95% uncertainty; actual paired power depends on discordant outcomes. Increase sample size before sealing if the development power analysis requires it.


## 14  Experiment matrix and decision rules

| ID | Comparison | Question answered |
| --- | --- | --- |
| B0 | Original 27B BF16 and supported low-bit version | How much capability and runtime are available without parameter compression? |
| B1 | Native Qwen3.5-2B before and after the same healing | Can a pretrained small model reach the objective more cheaply? |
| A0 | Direct 2.95B pruning + ordinary KD | How recoverable is the simplest inherited-weight student? |
| A1 | Staged pruning vs A0 | Do intermediate repairs justify their extra compute? |
| A2 | Activation-aware SVD vs dense A1 | Does factorization improve the quality/latency frontier? |
| N1 | Semantic allocation vs reconstruction-only | Does quantum failure sensitivity preserve important capacity? |
| N2 | Memory probes vs matched long-context KD | Do observable prefix probes repair hybrid memory? |
| N3 | Verified repair vs ordinary on-policy KD | Does error-specific supervision improve fresh code success? |
| R1 | SAPO or GRPO vs continued KD | Does verification RL add benefit after healing? |
| Q1 | PTQ vs QAH-style KD vs extra BF16 KD | Does quantization-aware recovery earn its cost? |

Run B0/B1/A0 first, then A1 if direct pruning has poor recovery or a clear staged hypothesis. Introduce N1–N3 individually after a stable reference exists. Do not run every combination. Only combine factors that independently survive their controls. DELTA-style joint rank/precision is admitted after A2 and a packed low-bit export work.


### Fairness has two views

Report equal effective training-token comparisons for scientific attribution, and equal total accelerator/teacher/verifier cost comparisons for practical choice. A staged path spends tokens on larger intermediate models; count them. Do not credit a treatment with extra data, longer generation or more teacher calls and attribute the gain solely to its algorithm. Native 2B and custom 2.95B models differ in size: report the size/cost frontier and add a closer-size pretrained control if useful. A same-shape randomly initialized student can test the value of inherited weights in a bounded ablation.


### Continue only when there is evidence

If early candidates overlap, continue the best two to the next fixed budget, because short healing can misrank architectures. If structural compression remains below native-small distillation after comparable total cost, prefer route B for delivery and retain route A as a failed or incomplete research result. If 3B fails but 7B works, report that result without calling it the requested 3B outcome.

Use development evaluations for selection and open sealed tests once per frozen confirmation round. A later revision needs a new independent sealed set or an explicitly declared reused-test limitation. Do not keep tuning to the same final benchmark.


## 15  Compute model and realistic schedule

The following are transparent planning calculations, not measured Ascend throughput. Online distillation has a substantial teacher cost even though the student is small. For dense parameter-dominated computation, use the first-order proxy below and separately measure attention, recurrent kernels, vocabulary projection, communication and generation overhead.

F_total ≈ sum over stages i of (6 P_student,i + 2 P_teacher,i) N_i

wall_seconds ≈ F_total / aggregate_sustained_FLOPs_per_second

With a 3B student and 27B teacher, the proxy is 72 billion FLOPs per training token. Ten billion tokens therefore cost roughly 7.2 × 10²⁰ FLOPs. Earlier 14B/7B stages cost more per token; autoregressive teacher generation and simulator time add work beyond this estimate.

| Training tokens | At 0.8 PFLOP/s sustained | At 1.6 PFLOP/s sustained |
| --- | --- | --- |
| 1B | 25 hours | 12.5 hours |
| 10B | 250 hours or 10.4 days | 125 hours or 5.2 days |
| 100B | 2500 hours or 104 days | 1250 hours or 52 days |

For illustration, 0.8 PFLOP/s could mean eight devices sustaining 100 TFLOP/s each; this is not a prediction for 910B/910C. At that assumption the 10B-token case uses 2000 aggregate device-hours before overhead. Actual elapsed time comes from a 200-step target-runtime benchmark and a representative generation/verifier queue. Do not budget using peak hardware specifications.


### Memory and storage

A conventional full-student Adam setup can require about 16 bytes per parameter for weights, gradients, master weights and optimizer states: around 47 GB for 2.95B before activations. Actual optimizers differ. Shard the student and separately serve or shard the teacher; test 2K, 8K and packed batches before booking a full run. Maintain space for two complete student training checkpoints, immutable teacher files, data and evaluation outputs.


### Milestones measured from a working environment

Weeks 1–2: native parity, evaluator audit, data pilot, B0/B1 and compressed-model lifecycle. Weeks 3–4: bounded architecture screening and the two recovery finalists. Weeks 5–8: the selected quantum healing experiment, on-policy study, quantized export and a pilot report, conditional on measured throughput. A 20B–100B or larger recovery extension may take additional months and requires an evidence-based allocation.

Minitron precedents range from tens of billions to roughly 380B recovery tokens at milder compression. Thus 1B–3B-token runs are feasibility evidence. Broad 9-fold recovery may require more than 100B tokens or fail at an economical budget. Reuse the original teacher directly for the final student; an intermediate assistant teacher can reduce cost but must be compared because it can pass on compression errors. [S03–S04]


## 16  Engineering work and failure recovery

Implement this study in a separate research namespace such as compression_3b. The paths below are proposed additions, not claims that runnable tools already exist. Reuse tested repository utilities only after checking their contracts against real language-model and multi-qubit data. [S19]

| Module | Required implementation and test |
| --- | --- |
| model_inventory and surgery | Module shapes, unique parameter count, consistent slices, identity reconstruction, export/reload. |
| calibration and rank_search | Training-only activations, stable covariance, dependency groups, rank/bit budgets, measured latency. |
| data_factory and verifier | Ancestry splits, SDK environments, hidden tests, known-good and mutant checks, executed-test receipts. |
| distill and repair | CE/KL masks, tokenizer parity, chunked logits, verified student-prefix repair, scalar/gradient loss tests. |
| rl and quantize | Pinned GRPO/SAPO, actor masks, reward isolation, fake-versus-packed parity, exported runtime probes. |
| evaluate and run_registry | Frozen contracts, per-task outcomes, uncertainty, cost ledger, transactional checkpoints and recovery. |


### Make long runs survive disconnection

Launch through a durable scheduler or service and record actual job IDs, hosts, ranks and device allocation. Monitors discover jobs from the registry, not from a naming-pattern regex. Browser or terminal disconnection must not terminate a healthy trainer. Use a separate supervisor with a heartbeat that proves completed optimizer steps, not just process existence.

Write checkpoints transactionally with weights, optimizer, scheduler, RNG, sampler, data cursor, curriculum, quantization observers, architecture and teacher hashes. Publish a completion marker only after all files verify. Test save/terminate/resume against an uninterrupted run, including next-batch identity, learning rate and parameter updates. Never recover from an incomplete directory merely because its timestamp is newest.


### Stop conditions that preserve useful evidence

Pause on NaN/Inf, inconsistent loss masks, disconnected gradients, corrupted checkpoints, verifier failures, sustained zero progress, unexpected data mixing or a measured retention breach. A proposed stall threshold is the larger of 15 minutes and five times the recent median step duration, adjusted for known checkpoint phases. Include monitor liveness so a dead monitor cannot silently report health.

Retry a transient transport failure once, then diagnose. Before relaunching, query the real job to avoid duplicate trainers. A code, model, loss or data change creates a child experiment. Save first-cause logs, resource state and the last complete checkpoint; continue automatically after a tested correction within the allocated budget.


## 17  First ten working days and deliverables

| When | Concrete outcome | Decision |
| --- | --- | --- |
| Days 1–2 | Pinned source model and loader; parameter inventory; native logits and cache parity; assigned-device benchmark. | Resolve runtime mismatches before surgery. |
| Days 3–4 | At least 200 independently checked training roots; evaluator mutant suite; initial teacher and native-small baselines. | Reject weak task contracts and broken reference environments. |
| Days 5–6 | 2.95B model construction, identity-surgery test, save/reload and real resumed optimizer steps. | Parameter cap and functioning training are mandatory. |
| Days 7–8 | 50M-token initial screens for the highest-value paths; recovery curves, output samples and resource accounting. | Advance only healthy candidates; avoid screening all variants blindly. |
| Days 9–10 | Paired development evaluation, cost projection and first repair-data slice. | Select the next two candidates or switch delivery priority to native-small distillation. |

This sequence is conditional on model access, working kernels and sufficient assigned compute. If a gate consumes the period, report the blocked gate and its evidence rather than manufacture a ten-day training result. The important early output is a defensible continuation decision.


### Release package

Deliver complete text-model weights and a tested packed derivative where supported; immutable source-model linkage; student configuration and module mapping; tokenizer/template; dependency and hardware locks; data lineage and split hashes; training and evaluation contracts; executable inference example; checkpoint/resume instructions; per-task outcomes; confidence intervals; failure taxonomy; and total compute including teacher generation and discarded candidates.

The model card distinguishes measured, estimated and unmeasured values. Include parameter count, exact bytes, tested context lengths, native and tool-enabled performance, SDK versions and known weak skills. Provide a clean fresh-process load and a reproducible quantum-code example whose semantics are independently verified.


### Main risks and response

Irrecoverable capacity loss: shrink more gradually or choose native-small distillation. Broad forgetting: increase replay and reduce specialization rate. Teacher errors: prefer executable references and verified alternative solutions. Reward sparsity: add warm-start healing and easier training tasks. Poor kernel efficiency: choose regular dense shapes or higher precision. Data leakage: rebuild root-disjoint splits. Unreliable runs: repair the supervisor and resume path before spending more compute.


### Research claim to aim for

A credible paper would show that a specific semantic allocation or verified recovery method improves fresh quantum-code correctness at a fixed parameter, training and inference budget, with independent seeds and ablations. It would also show the cost of reaching that result and where the method fails. Achieving 100% on a familiar benchmark is an optional aspiration, not the definition of scientific success.


## 18  Primary sources and project references

Sources were checked for this plan on 16 November 2026. Source summaries identify published mechanisms and limitations; numerical budgets and candidate architectures elsewhere in this document are proposed or analytically derived. Mutable model and software sources must be pinned again when a run contract is resolved.

S01  Official Qwen3.8-27B configuration Open source

S02  Transformers Qwen3.5 model implementation Open source

S03  Compact Language Models via Pruning and Knowledge Distillation 2024 Open source

S04  Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning 2025 Open source

S05  SVD-LLM V2 2025 Open source

S06  DELTA 2026 Open source

S07  MiniLLM 2023 Open source

S08  DistiLLM-2 2025 Open source

S09  GKD implementation and on-policy distillation documentation Open source

S10  Quantization-Aware Healing 2026 author report Open source

S03 data-volume cross-check: the official Minitron-4B model card reports 94B continued-training tokens. S18 was available through its indexed arXiv abstract; implementation details of that new audit should be inspected before reproducing it. No numerical results from the audit are used as project results.


## 19  Source scope and reproducibility notes

S11  AWQ Open source

S12  SpinQuant Open source

S13  QLoRA Open source

S14  Soft Adaptive Policy Optimization Open source

S15  Qwen3.5 quantization examples in LLM Compressor Open source

S16  vLLM Ascend quantization documentation Open source

S17  Qiskit HumanEval source repository Open source

S18  Benchmarking the Quantum Leap 2026 preprint Open source

S19  Existing quantum-computing-gpt governing plan Open source

S20  Official Qwen3.5-2B configuration Open source


### Evidence scope

The QAH author report motivates comparison with the original teacher; its reported gains concern a particular structurally compressed GPT-OSS pipeline, not this 9-fold Qwen reduction. Extra distillation can confound comparisons against an earlier BF16 checkpoint, so the plan includes the continued-BF16-KD control. No published stability or speedup claim is assumed to transfer automatically.

Minitron-SSM uses a Mamba-based hybrid. Its dependence-aware pruning principle is relevant, but DeltaNet has different gates and state updates. The proposed surgery and memory probes therefore require new implementation and validation. Likewise, current Qwen-family quantization support is not evidence that every custom student shape trains or serves efficiently on the assigned Ascend devices.

Repository review observed README blob 69b0fc88f443817f36438cca47385764a8de0bd4 and source tree 58d48a1e328c174607095851518fa424e4d5e14b. The inspected moving Transformers implementation had blob 0262331cd91772b3fcbc3ba3fe72b8fa2e20d71f. These identify the reviewed evidence; they are not a tested runtime lock. Resolve commit IDs and archive exact dependencies before execution.


### Unmeasured items

No Qwen model was loaded, compressed, trained or benchmarked in preparing this plan. The 2.94949B count and storage/FLOP estimates are arithmetic under explicit module assumptions. Runtime support, recovery quality, required training tokens, scientific novelty and achievable speed remain experimental questions. The first two weeks are designed to resolve the cheapest and most consequential of them.

