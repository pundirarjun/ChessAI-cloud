from collections import deque
import random
from collections import Counter


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

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def clear(self):
        self.buffer.clear()
