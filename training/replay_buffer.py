from collections import deque
import random
from collections import Counter
import math

import numpy as np


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, samples, source_iteration=None):
        """Add samples and optionally tag them with their RL iteration.

        Samples are stored as 4-tuples: (state, policy, value, iteration).
        Existing 3-tuples remain valid and are treated as legacy/unknown.
        """
        if source_iteration is None:
            self.buffer.extend(samples)
            return

        tagged = []
        for sample in samples:
            if len(sample) >= 4:
                tagged.append(sample)
            else:
                tagged.append((sample[0], sample[1], sample[2], int(source_iteration)))
        self.buffer.extend(tagged)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError("Not enough samples in replay buffer.")
        return random.sample(self.buffer, batch_size)

    def iteration_distribution(self):
        """Return counts of samples by their source RL iteration."""
        counts = Counter()
        for sample in self.buffer:
            if len(sample) >= 4:
                counts[int(sample[3])] += 1
            else:
                counts["unknown"] += 1
        return counts

    def print_iteration_distribution(self):
        counts = self.iteration_distribution()
        total = len(self.buffer)

        print("\n============================================================")
        print("REPLAY BUFFER DISTRIBUTION")
        print("============================================================")
        print(f"Total samples: {total:,}")

        if not counts:
            print("Buffer is empty.")
        else:
            known = [(k, v) for k, v in counts.items() if k != "unknown"]
            known.sort(key=lambda x: x[0])
            for iteration, count in known:
                pct = (count / total * 100.0) if total else 0.0
                print(f"RL{iteration}: {count:>8,} ({pct:6.2f}%)")
            if "unknown" in counts:
                count = counts["unknown"]
                pct = (count / total * 100.0) if total else 0.0
                print(f"UNKNOWN: {count:>7,} ({pct:6.2f}%)")

        print("============================================================\n")

    def validate_provenance(
        self,
        *,
        expected_max_iteration=None,
        sample_limit=2048,
    ):
        """Validate a bounded, deterministic sample without changing replay data.

        Historical three-tuples remain supported and are reported as unknown;
        tagged samples may not claim to come from a future iteration.
        """
        total = len(self.buffer)
        if total == 0:
            raise RuntimeError("Replay buffer is empty.")

        distribution = self.iteration_distribution()
        known_iterations = [key for key in distribution if key != "unknown"]
        if expected_max_iteration is not None:
            future = [i for i in known_iterations if i > int(expected_max_iteration)]
            if future:
                raise RuntimeError(
                    "Replay buffer contains samples newer than its expected "
                    f"checkpoint: {sorted(future)} > RL{expected_max_iteration}."
                )

        # Evenly spaced indices make this deterministic and include old/new FIFO
        # regions without allocating or shuffling the whole replay buffer.
        checks = min(int(sample_limit), total)
        indices = sorted({(i * (total - 1)) // max(1, checks - 1) for i in range(checks)})
        for index in indices:
            sample = self.buffer[index]
            if len(sample) not in (3, 4):
                raise RuntimeError(f"Replay sample {index} has invalid tuple length {len(sample)}.")
            state, policy, value = sample[:3]
            state_array = np.asarray(state)
            policy_array = np.asarray(policy)
            if state_array.shape != (18, 8, 8):
                raise RuntimeError(
                    f"Replay sample {index} has state shape {state_array.shape}, expected (18, 8, 8)."
                )
            if policy_array.shape != (4544,):
                raise RuntimeError(
                    f"Replay sample {index} has policy shape {policy_array.shape}, expected (4544,)."
                )
            if not np.isfinite(state_array).all() or not np.isfinite(policy_array).all():
                raise RuntimeError(f"Replay sample {index} contains non-finite state or policy values.")
            if np.any(policy_array < -1e-6) or not np.isclose(
                float(policy_array.sum()), 1.0, atol=1e-4, rtol=1e-4
            ):
                raise RuntimeError(f"Replay sample {index} has an invalid policy target.")
            value_float = float(value)
            if not math.isfinite(value_float) or value_float not in (-1.0, 0.0, 1.0):
                raise RuntimeError(f"Replay sample {index} has invalid value target {value!r}.")

        return {
            "total_samples": total,
            "checked_samples": len(indices),
            "source_distribution": dict(distribution),
            "unknown_samples": int(distribution.get("unknown", 0)),
        }

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def clear(self):
        self.buffer.clear()
