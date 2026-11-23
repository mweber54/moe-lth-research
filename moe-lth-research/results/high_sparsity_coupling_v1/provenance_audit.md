# Cohort and provenance audit

Frozen cohort SHA256: `d7162755c0787a3b6d545940e7129954fb8f429987619ec1a0581e8013366082`. Frozen protocol SHA256: `49fe4fea5da5a145d6364c5cae9a8ee63d831a52b54286f77511a8453a669bdd`.

Ten sparse conditions and five distinct matched dense recoveries; 120 checkpoint-role records and 307 registered files verified. 45 IMP discovery stages across five seeds.

Each .80 mask is selected from round_07; each .85 mask from round_08. All nine rounds restart at original initialization, with fresh optimizer and trainable R0 router. The final sparse conditions restart separately with those exact masks. Dense comparators are canonical R0 dense recovery artifacts, not source-final training checkpoints. The source-final dense is only an additional reference.

IMP discovery applied updates range 2497–2499 of 2500 attempts. Therefore attempted steps are not interchangeable with applied updates.

Full historical recovery states are available at 2500,3000,4000,5000. Recovery initialization is reconstructable exactly from verified original init and fixed mask, checked against saved component hashes. Early historical recovery intermediates are absent; router-only snapshots cannot replace them. Original source training has its own intermediate checkpoints; those are not recovery intermediates.

45 checkpoint-role records contain optimizer state; 45 contain scaler state; 45 contain all four RNG families. The 2500-step model-only boundary deliberately resets AdamW/scaler for BOTH sparse and dense continuations. Full extension checkpoints record training_data_position; no live sampler object was saved. Native materialized batches plus position and hashes define deterministic ordering.

Training: fixed 128 sequences ×128 byte tokens, 5000 attempted steps, 81920000 token presentations. Validation: original 32 materialized batches, sequence identity/hash checked. Validation participates in historical ticket selection and is not independent test. No demonstrably unused test split is established.

Historical strict validation successes: 4/10 over 3 source seeds, with related observations at two sparsities for seed 7. Continuous gaps retained:

```csv
seed,sparsity,sparse_ce,dense_ce,gap,strict_success,label_origin,checkpoint_sha256
7,0.8,1.3295407060960809,1.365141073325593,-0.0356003672295124,True,historical_validation,7b2a7fe18b2a80baa68ed39991b5a5c7c973f11a1b91e5d26025ff415e0a5501
7,0.85,1.343936258270922,1.365141073325593,-0.0212048150546713,True,historical_validation,0e44809db4ef2bac5fd0c8d096dbf31c47fe305b1e6a4d58fadc7d9728c0bddb
17,0.8,1.3510094219472522,1.3298924914390535,0.0211169305081984,False,historical_validation,334689157a2bcc0137f732f7f67e8da0098a7732cef301531b055a8362e874cd
17,0.85,1.3661630529845443,1.3298924914390535,0.0362705615454908,False,historical_validation,8ea3fa0d138352748856b041487730b8918dabf5cc8b7b4d754da8a576c9b004
29,0.8,1.3451853395902804,1.3413966347218675,0.0037887048684128,False,historical_validation,76862b9618b052feffbcca0d5bcfeb7b2f04f27c3d6421400442a5b83431d929
29,0.85,1.3661939647249164,1.3413966347218675,0.0247973300030488,False,historical_validation,85e3c76475a8c9762b59d29346cd7142a94c2851c903cbf6b8a66364d483e7c8
43,0.8,1.3377270578315483,1.3394907390355506,-0.0017636812040022,True,historical_validation,885bf68588d1a6667989e59017d185091e7af56406f44978fc03d3ffc7918075
43,0.85,1.3526891602272488,1.3394907390355506,0.0131984211916982,False,historical_validation,7f22319bd729d3d04ffaa070487e1206b3a4c55452de8a2faead7767b61a676b
61,0.8,1.3392723979409529,1.3661851543310497,-0.026912756390097,True,historical_validation,873d00d6da0e038e3a9c781f0f1412039ce2a4dd215c26fd61ba529337cc3782
61,0.85,1.3957522647362857,1.3661851543310497,0.0295671104052359,False,historical_validation,191a440bb38633f066b4b4d4094a3f4b88e1021151739042a16d2713b74727a1
```

Full code/source history of the original source trainer was inspected; original source model-only checkpoints do not establish AMP skipped-update counts or exact optimizer/RNG continuation. Do not infer unrecorded source update counts. Source training is not rerun.

Both-host verification results are stored in hosts/. GPU benchmark and validation results are separate launch gates. No historical directories are modified.
