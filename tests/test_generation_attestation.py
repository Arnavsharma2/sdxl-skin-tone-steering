import json

from generate_training_data import (
    append_generation_attestation,
    configured_seed_schedule,
    load_generation_attestations,
    valid_generation_attestation,
)


def test_generation_attestation_is_content_addressed(tmp_path):
    image_path = tmp_path / "portrait.png"
    ledger_path = tmp_path / "campaign.jsonl"
    image_path.write_bytes(b"generated-image")
    expected = {
        "group": "light",
        "index": 0,
        "path": str(image_path),
        "descriptor": "light skin tone",
        "seed": 42,
        "generation_signature": "a" * 64,
    }

    append_generation_attestation(image_path, expected, ledger_path)

    assert valid_generation_attestation(image_path, expected, ledger_path)
    record = load_generation_attestations(ledger_path)[str(image_path)]
    assert record["schema_version"] == "1.0"
    assert len(record["sha256"]) == 64
    json.loads(ledger_path.read_text())

    image_path.write_bytes(b"tampered-image")
    assert not valid_generation_attestation(image_path, expected, ledger_path)


def test_generation_attestation_rejects_changed_campaign_inputs(tmp_path):
    image_path = tmp_path / "portrait.png"
    ledger_path = tmp_path / "campaign.jsonl"
    image_path.write_bytes(b"generated-image")
    expected = {
        "group": "dark",
        "index": 3,
        "path": str(image_path),
        "descriptor": "deep skin tone",
        "seed": 512,
        "generation_signature": "b" * 64,
    }
    append_generation_attestation(image_path, expected, ledger_path)

    changed = {**expected, "seed": 513}
    assert not valid_generation_attestation(image_path, changed, ledger_path)


def test_configured_seed_schedule_prefers_frozen_data_seeds():
    class Config:
        data = {"seed_schedule": [1001, 2002, 3003]}

    assert configured_seed_schedule(Config()) == [1001, 2002, 3003]
