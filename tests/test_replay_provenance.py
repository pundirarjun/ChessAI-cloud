import numpy as np

from training.replay_buffer import ReplayBuffer


def _sample(value=0.0):
    policy = np.zeros(4544, dtype=np.float32)
    policy[0] = 1.0
    return np.zeros((18, 8, 8), dtype=np.float32), policy, value


def test_validate_provenance_accepts_legacy_and_tagged_samples():
    replay = ReplayBuffer(10)
    replay.add([_sample(0.0)])
    replay.add([_sample(1.0)], source_iteration=53)

    report = replay.validate_provenance(expected_max_iteration=53)

    assert report["total_samples"] == 2
    assert report["unknown_samples"] == 1
    assert report["source_distribution"][53] == 1


def test_validate_provenance_rejects_future_tag_and_invalid_policy():
    future = ReplayBuffer(10)
    future.add([_sample(-1.0)], source_iteration=54)
    try:
        future.validate_provenance(expected_max_iteration=53)
    except RuntimeError as error:
        assert "newer" in str(error)
    else:
        raise AssertionError("Expected future source tag to be rejected.")

    invalid = ReplayBuffer(10)
    state, policy, value = _sample()
    policy[0] = 0.5
    invalid.add([(state, policy, value)])
    try:
        invalid.validate_provenance()
    except RuntimeError as error:
        assert "policy" in str(error)
    else:
        raise AssertionError("Expected invalid policy target to be rejected.")
