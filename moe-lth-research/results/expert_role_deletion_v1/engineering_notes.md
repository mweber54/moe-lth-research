# Engineering notes

The initial preparation attempt stopped before model diagnostics because probe
construction also rejected adjacent windows inside the same complete batch and
searched only 16 batches. No routing/damage outcome had been evaluated. The rule
was corrected to allow internal batch adjacency while requiring disjoint raw
windows (including the shifted target boundary) between retained complete batches.
The deterministic search upper bound became 512 native batches; the first four
eligible batches were 0, 1, 2, and 18. The frozen A uses 0/1, B uses 2/18.
This changes neither sequence structure nor native capacity context. No group,
expert, checkpoint, threshold or deletion dose was selected using effects.

The initial inventory named two absent conventional filenames. Inspection located
imp_mask_information_v1/protocol.md and the replay repair corrected_interpretation.md;
the inventory was corrected before the measurement implementation was sealed.

The GPU environment lacks pytest; safeguards ran successfully in the existing
.venv environment. Production measurement used .venv-highres. Six safeguard tests
passed. No package installation or historical artifact edits were needed.
