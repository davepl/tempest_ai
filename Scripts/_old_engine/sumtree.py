import numpy as np

class SumTree:
    """
    SumTree data structure for Prioritized Experience Replay.
    Stores priorities in a binary tree for O(log N) sampling and updating.
    """
    def __init__(self, capacity):
        self.capacity = int(capacity)
        # Tree size: 2 * capacity - 1
        # We use a fixed size array.
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.write = 0
        self.count = 0

    def add(self, p):
        """
        Add a new priority.
        The data index corresponds to self.write.
        """
        tree_idx = self.write + self.capacity - 1
        self.update(tree_idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.count < self.capacity:
            self.count += 1

    def update(self, tree_idx, p):
        """
        Update priority at tree_idx.
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def update_by_data_idx(self, data_idx, p):
        """
        Update priority given the data index (0 to capacity-1).
        """
        tree_idx = data_idx + self.capacity - 1
        self.update(tree_idx, p)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        """
        Retrieve data index and priority for a given value s.
        Returns (tree_idx, priority, data_idx)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], dataIdx)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]
    
    def max_p(self):
        # This is an approximation or requires another tree (MaxTree)
        # For PER, we usually track max_p separately in the buffer
        pass
