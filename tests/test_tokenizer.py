from nanovlm.data.tokenizer import NanoVLMTokenizer, train_tokenizer

DUMMY_CORPUS = [
    "the cat sits on the mat",
    "a dog runs in the park",
    "the cat and the dog play together",
    "a big red ball rolls down the hill",
    "two kids laugh and jump near the tree",
] * 20  # BPE needs enough repetition to learn merges from a tiny vocab


def test_train_and_roundtrip(tmp_path):
    train_tokenizer(DUMMY_CORPUS, vocab_size=300, out_dir=str(tmp_path))
    tok = NanoVLMTokenizer.from_dir(str(tmp_path))

    assert tok.vocab_size <= 300
    for special_id in (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id):
        assert special_id is not None

    text = "the cat sits on the mat"
    ids = tok.encode(text)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id

    decoded = tok.decode(ids)
    assert decoded.strip() == text


def test_encode_without_bos_eos(tmp_path):
    train_tokenizer(DUMMY_CORPUS, vocab_size=300, out_dir=str(tmp_path))
    tok = NanoVLMTokenizer.from_dir(str(tmp_path))

    ids = tok.encode("a dog runs", add_bos=False, add_eos=False)
    assert tok.bos_id not in ids
    assert tok.eos_id not in ids
