# Fixed recovery-trajectory schedule intervention

Six native R100 dense recoveries capture every actual training forward and every historical validation forward. All recipient router ages, and both sparse/dense models, replay the corresponding architecture/seed schedule. Replayed IDs, capacity priorities, and accepted masks are checked against the donor on every forward; complete chronological archive-hash sequences must match. Recipient gate amplitudes remain differentiable and router parameters trainable.

Primary endpoint: both existing native-trained and new clamped-trained models are evaluated using the identical donor validation routes. Native evaluation of both arms is secondary, separating evaluation-policy changes from training effects. No extra native baseline recoveries are run beyond the six necessary schedule donors. Donor capture must reproduce history, and R100 dense self-replay must reproduce the donor exactly before other ages run.

The previous grid, E0/shared/router hashes, ET-derived 85% masks, batch sequences, seeds, fresh AdamW hyperparameters, no LR schedule, fp16 training, clipping, GradScaler, masking, and 2500-step budget are retained. Nine recoverable snapshots are saved. Native routing-stat probes are non-mutating measurements of router proposals; they are not used for parameter updates. Actual update routing is defined by the replay audit.

Bulk data are stored through symlinks to the dedicated local /tmp directory recorded in run_manifest.json because the historical-results filesystem is nearly full. Historical artifacts are never changed.
