"""Native order-study job executor with sealed treatment assembly substituted.

Copied before sealing. The only scientific change is loading the registered
surviving-value variant instead of reconstructing IMP-original coordinates.
"""
from run import native, variant_state, OUT
# Reuse native imported helpers without changing historical source files.
globals().update({k:v for k,v in vars(native).items() if not k.startswith('__')})
from run import OUT, variant_state

def _run_one_job(row: dict, resume: bool = False, device_index: int = 0) -> None:
    job_id = row["job_id"]
    lock_path = OUT / "locks" / f"{job_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        previous = _job_status_path(job_id)
        previous_status = _json(previous).get("status") if previous.is_file() else None
        if previous_status == "COMPLETE":
            return
        resume_attempt = resume or previous_status == "RUNNING"
        attempt, resuming = _new_or_resumable_attempt(job_id, resume_attempt)
        _write_status(
            job_id,
            status="RUNNING",
            attempt=str(attempt),
            host=socket.getfqdn(),
            pid=os.getpid(),
            resumed=resuming,
            error=None,
        )
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("Production recovery requires the verified CUDA FP16 environment")
            if not torch.__version__.startswith("2.7.0"):
                raise RuntimeError(f"Expected torch 2.7.0, found {torch.__version__}")
            device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device)
            config, canonical_train, validation = _load_training_material(int(row["seed"]))
            if not config["training"].get("precision", "").lower().startswith("fp16"):
                raise RuntimeError("Source training precision does not match the audited CUDA FP16 protocol")
            permutations = _permutations()[row["order"]]
            ordered_indices = np.concatenate(
                (permutations["segment_0"], permutations["segment_1"])
            )
            original_hashes = _seed_entry(int(row["seed"]))["per_batch_content_hashes"]
            permuted_hashes = [original_hashes[int(index)] for index in ordered_indices]
            if Counter(permuted_hashes[:SEGMENT_SIZE]) != Counter(original_hashes[:SEGMENT_SIZE]):
                raise RuntimeError("Permutation changed the first segment batch multiset")
            if Counter(permuted_hashes[SEGMENT_SIZE:]) != Counter(original_hashes[SEGMENT_SIZE:]):
                raise RuntimeError("Permutation changed the second segment batch multiset")
            first_batches = _apply_order(canonical_train, permutations["segment_0"])
            second_batches = _apply_order(canonical_train, permutations["segment_1"])
            initial_path = Path(row['initial_checkpoint'])
            initial_state, masks = variant_state(row)
            init_hash = _tensor_state_hash(initial_state)
            source_snapshot = attempt / "snapshots/step_0000.pt"
            if not source_snapshot.exists():
                initial_model = assemble_router_age_model(
                    config["model"], initial_state, str(initial_path), masks, device
                )
                _assert_model_mask(initial_model, masks)
                _model_snapshot(source_snapshot, initial_model, 0, config, "production_model_only")
                del initial_model
            first_output = attempt / "segment_0"
            first_checkpoint = first_output / "checkpoints/final_recovered.pt"
            if first_checkpoint.is_file() and (first_output / "segment_record.json").is_file():
                first_state = _read_model_payload(first_checkpoint)["model"]
                first_record = _json(first_output / "segment_record.json")
            else:
                first_record = _run_segment(
                    row,
                    attempt,
                    config,
                    initial_state,
                    masks,
                    initial_path,
                    first_batches,
                    validation,
                    0,
                    "segment_0",
                    device,
                )
                first_state = _read_model_payload(first_checkpoint)["model"]
            boundary_reset = _save_boundary_reset(
                attempt, config, int(row["seed"]), first_checkpoint, first_state, masks
            )
            if first_output.joinpath("interruption.pt").is_file():
                first_interrupt = torch.load(
                    first_output / "interruption.pt", map_location="cpu", weights_only=False
                )
                boundary_reset["segment1_optimizer_state_entries_before_reset"] = len(
                    first_interrupt.get("optimizer", {}).get("state", {})
                )
                boundary_reset["segment1_scaler_state_before_reset"] = first_interrupt.get("scaler", {})
                _atomic_json(attempt / "reset_boundary.json", boundary_reset)
            second_output = attempt / "segment_2500"
            second_checkpoint = second_output / "checkpoints/final_recovered.pt"
            if second_checkpoint.is_file() and (second_output / "segment_record.json").is_file():
                second_record = _json(second_output / "segment_record.json")
            else:
                second_record = _run_segment(
                    row,
                    attempt,
                    config,
                    first_state,
                    masks,
                    first_checkpoint,
                    second_batches,
                    validation,
                    SEGMENT_SIZE,
                    "segment_2500",
                    device,
                )
            snapshot_list = _snapshot_paths(attempt)
            actual_stream_hash = recovery._batch_sequence_hash(canonical_train)
            eval_rows = _common_evaluate(
                snapshot_list,
                config,
                validation,
                int(row["seed"]),
                row["order"],
                row["model_type"],
                job_id,
                _seed_entry(int(row["seed"]))["validation_batch_sequence_hash"],
                device,
            )
            eval_path = attempt / "common_evaluations.jsonl"
            with eval_path.open("w", encoding="utf-8") as stream:
                for item in eval_rows:
                    stream.write(json.dumps(item, sort_keys=True) + "\n")
            updates = _update_accounting(attempt)
            if len(updates) != TOTAL_STEPS:
                raise RuntimeError(f"Expected 5,000 attempted updates for {job_id}, found {len(updates)}")
            if sum(int(item["applied"]) for item in updates) > TOTAL_STEPS:
                raise RuntimeError(f"Applied-update accounting exceeds attempts for {job_id}")
            if not boundary_reset["model_parameters_unchanged"]:
                raise RuntimeError(f"Parameters changed at the fresh-optimizer reset for {job_id}")
            boundary_updates = _boundary_update_rows(updates)
            if {item["step"] for item in boundary_updates} != {2499, 2500, 2501, 2525}:
                raise RuntimeError(f"Applied-update accounting around reset is incomplete for {job_id}")
            _atomic_json(
                attempt / "update_boundary_accounting.json",
                {
                    "boundary_step": 2500,
                    "updates": boundary_updates,
                    "segment1_optimizer_state_entries": boundary_reset.get("segment1_optimizer_state_entries_before_reset"),
                    "segment1_scaler_state": boundary_reset.get("segment1_scaler_state_before_reset"),
                    "segment2_fresh_scaler_state": boundary_reset["scaler_probe_state"],
                    "segment2_optimizer_state_entries_before_first_update": boundary_reset["fresh_optimizer_state_entries"],
                },
            )
            if [item["step"] for item in eval_rows] != list(SNAPSHOT_STEPS):
                raise RuntimeError(f"Common evaluation snapshots incomplete for {job_id}")
            reset_path = attempt / "reset_boundary.json"
            status = {
                "status": "COMPLETE",
                "attempt": str(attempt),
                "attempted_updates": len(updates),
                "applied_updates": sum(int(item["applied"]) for item in updates),
                "amp_skipped_updates": sum(int(item["amp_skipped"]) for item in updates),
                "step_2500_model_reset_hash_equal": True,
                "common_evaluation_steps": [item["step"] for item in eval_rows],
                "common_evaluation_target_tokens": eval_rows[-1]["target_tokens"],
                "common_evaluation_sha256": _sha256_file(eval_path),
                "initial_masked_state_hash": init_hash,
                "permuted_stream_hash_segment_0": recovery._batch_sequence_hash(first_batches),
                "permuted_stream_hash_segment_1": recovery._batch_sequence_hash(second_batches),
                "original_stream_hash": actual_stream_hash,
                "segment_0_seconds": first_record["segment_seconds"],
                "segment_1_seconds": second_record["segment_seconds"],
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pilot_checks": None,
            }
            _atomic_json(attempt / "completion.json", status)
            _write_status(job_id, **status)
        except BaseException as error:
            _write_status(
                job_id,
                status="FAILED",
                attempt=str(attempt),
                host=socket.getfqdn(),
                pid=os.getpid(),
                error=repr(error),
                failed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            raise

