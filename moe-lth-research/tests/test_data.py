from moe_lth.data import ByteNgramTokenizer, ByteTokenizer, load_token_splits


def test_byte_ngram_tokenizer_round_trips_and_compresses_repeated_text():
    text = "banana bandana banana bandana\n" * 8
    tokenizer = ByteNgramTokenizer.train(
        text,
        vocab_size=270,
        max_ngram=8,
        train_bytes=1000,
        min_frequency=2,
    )

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    assert max(token_ids) >= ByteTokenizer.vocab_size
    assert len(token_ids) < len(ByteTokenizer.encode(text))


def test_subword_token_splits_use_cache(tmp_path):
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    cache_path = tmp_path / "byte_ngram_vocab.json"
    train_path.write_text("alpha beta gamma alpha beta gamma\n" * 20, encoding="utf-8")
    validation_path.write_text("gamma beta alpha\n" * 5, encoding="utf-8")
    data_config = {
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_fraction": 0.9,
        "tokenizer": "byte_ngram",
        "tokenizer_vocab_size": 300,
        "tokenizer_cache_path": str(cache_path),
        "tokenizer_train_bytes": 1000,
        "tokenizer_max_ngram": 8,
        "tokenizer_min_frequency": 2,
    }

    first_train, first_validation = load_token_splits(data_config)
    second_train, second_validation = load_token_splits(data_config)

    assert cache_path.exists()
    assert first_train == second_train
    assert first_validation == second_validation
    assert max(first_train) < int(data_config["tokenizer_vocab_size"])
    assert max(first_validation) < int(data_config["tokenizer_vocab_size"])
