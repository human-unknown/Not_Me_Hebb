"""
striatum.py — 纹状体 (Striatum)  [v6.0 完整实现]

Corresponding brain areas: Caudate + Putamen + Nucleus Accumbens (NAc)
Hierarchy: Cerebrum → Basal Ganglia → Striatum

DMS (Dorsomedial Striatum = Caudate): goal-directed behavior, A→O learning
DLS (Dorsolateral Striatum = Putamen posterior): habit behavior, S→R learning
NAc (Nucleus Accumbens): incentive salience, dopamine gating

Functions:
  - D1 direct pathway (Go):  sensory state → promote action (DA↑ → strengthen)
  - D2 indirect pathway (No-Go): sensory state → suppress action (DA↓ → strengthen)
  - Habit automation: repeated S→R mappings transfer from DMS to DLS
  - Devaluation diagnostic: detect if behavior has become habitual

Storage mechanism:
  Unlike hippocampal episodic memory, the striatum does NOT use ClusterNetwork.
  Procedural memory uses an S-R binding table — a sparse dict mapping.
  This corresponds to the anatomy of the basal ganglia: MSN (medium spiny neuron)
  synaptic weight matrices, not cell assemblies.

Reference:
  - Yin & Knowlton (2006). The role of the basal ganglia in habit formation.
  - Graybiel, A. M. (2008). Habits, rituals, and the evaluative brain.
"""

import numpy as np
from typing import Optional, Tuple, Dict


def _state_hash(state_vec: np.ndarray, n_bits: int = 12) -> int:
    """Hash sensory state to integer key.

    Uses sign bits of first n_bits dimensions → n_bits-bit integer.
    Provides coarse but discriminative state discretization.
    """
    if len(state_vec) < n_bits:
        padded = np.zeros(n_bits, dtype=np.float32)
        padded[:len(state_vec)] = state_vec[:len(state_vec)]
        state_vec = padded
    bits = (state_vec[:n_bits] > 0).astype(int)
    return int(sum(bits[i] << i for i in range(n_bits)))


class Striatum:
    """Striatum — procedural memory and habit learning.

    Two parallel pathways:
      D1 (Direct/Go):   state → promote action, DA↑ → strengthen
      D2 (Indirect/NoGo): state → suppress action, DA↓ → strengthen

    Habits gradually transfer from DMS (flexible, goal-directed)
    to DLS (rigid, automatic).
    """

    def __init__(self, n_actions: int = 5, d1_d2_balance: float = 0.5):
        self.n_actions = n_actions
        self.d1_d2_balance = d1_d2_balance  # 0=D2 dominant, 1=D1 dominant

        # D1 direct pathway: {state_hash → np.array(n_actions)} — action promotion strengths
        self.d1_weights: Dict[int, np.ndarray] = {}

        # D2 indirect pathway: {state_hash → np.array(n_actions)} — action suppression strengths
        self.d2_weights: Dict[int, np.ndarray] = {}

        # Habit strength tracking: {state_hash → float [0,1]} — habitization level
        self.habit_strength: Dict[int, float] = {}

        # State-action co-occurrence counts: {(state_hash, action) → int}
        self.cooccurrence: Dict[Tuple[int, int], int] = {}

        # Stats
        self.n_learn_events: int = 0
        self.n_habitual_actions: int = 0
        self._habit_ema: float = 0.0  # Global habitization EMA

    # ================================================================
    # Learning
    # ================================================================

    def learn(self, state_vec: np.ndarray, action: int,
              da_signal: float, reward: float = 0.0,
              automation_rate: float = 0.05):
        """One trial-and-error learning event.

        Args:
            state_vec: sensory state (D,) or (N,)
            action: executed action index [0, n_actions-1]
            da_signal: dopamine signal [-1, 1] — positive=RPE>0, negative=RPE<0
            reward: environmental reward [0, 1]
            automation_rate: habit automation speed
        """
        sh = _state_hash(state_vec)
        self.n_learn_events += 1

        # Ensure weight arrays exist
        if sh not in self.d1_weights:
            self.d1_weights[sh] = np.zeros(self.n_actions, dtype=np.float32)
        if sh not in self.d2_weights:
            self.d2_weights[sh] = np.zeros(self.n_actions, dtype=np.float32)

        # ---- D1 Direct Pathway (Go): DA↑ → strengthen action ----
        if da_signal > 0:
            # D1 MSN: dopamine via D1 receptors enhances direct pathway
            lr_d1 = 0.1 * da_signal * self.d1_d2_balance
            self.d1_weights[sh][action] += lr_d1 * (
                1.0 - self.d1_weights[sh][action])
        else:
            # DA↓ → D1 pathway slight decay
            self.d1_weights[sh][action] *= 0.98

        # ---- D2 Indirect Pathway (No-Go): DA↓ → suppress action ----
        if da_signal < 0:
            # D2 MSN: decreased dopamine disinhibits D2 → strengthens indirect pathway
            lr_d2 = 0.1 * abs(da_signal) * (1.0 - self.d1_d2_balance)
            self.d2_weights[sh][action] += lr_d2 * (
                1.0 - self.d2_weights[sh][action])
        else:
            # DA↑ → D2 pathway decays (dopamine inhibits D2 MSNs)
            self.d2_weights[sh][action] *= 0.98

        # ---- Co-occurrence count (for habit strength calculation) ----
        key = (sh, action)
        self.cooccurrence[key] = self.cooccurrence.get(key, 0) + 1

        # ---- Update habit strength ----
        # Habit strength = proportion of times this action was chosen in this state
        total_actions_in_state = sum(
            self.cooccurrence.get((sh, a), 0)
            for a in range(self.n_actions))
        if total_actions_in_state > 0:
            cooc = self.cooccurrence.get(key, 0)
            # Habitization = proportion of this action + repetition factor
            action_ratio = cooc / max(total_actions_in_state, 1)
            repetition_factor = 1.0 - np.exp(-total_actions_in_state / 20.0)
            new_habit = action_ratio * repetition_factor

            if sh not in self.habit_strength:
                self.habit_strength[sh] = new_habit
            else:
                # EMA update
                self.habit_strength[sh] += automation_rate * (
                    new_habit - self.habit_strength[sh])

        # Global habitization EMA
        if self.habit_strength:
            self._habit_ema += 0.01 * (
                np.mean(list(self.habit_strength.values())) - self._habit_ema)

    # ================================================================
    # Action Selection
    # ================================================================

    def get_habit_action(self, state_vec: np.ndarray,
                         novelty: float = 0.0) -> Optional[int]:
        """Select action from habit (DLS).

        Args:
            state_vec: sensory state
            novelty: current novelty signal [0,1] — high novelty suppresses habit

        Returns:
            Habit action index, or None (no habit)
        """
        sh = _state_hash(state_vec)
        hs = self.habit_strength.get(sh, 0.0)

        # Novelty suppresses habit (high novelty → need flexible behavior)
        effective_habit = hs * (1.0 - novelty * 0.7)

        # Habit not strong enough for automation
        if effective_habit < 0.3:
            return None

        if sh not in self.d1_weights or sh not in self.d2_weights:
            return None

        # Go - NoGo net value = action tendency
        net_strength = (self.d1_weights[sh]
                       - self.d2_weights[sh] * (1.0 - self.d1_d2_balance))

        # Select action with highest net strength
        best_action = int(np.argmax(net_strength))
        if net_strength[best_action] < 0.1:
            return None

        self.n_habitual_actions += 1
        return best_action

    def get_goal_directed_action(self, state_vec: np.ndarray) -> np.ndarray:
        """Get DMS goal-directed action tendency distribution.

        Returns:
            (n_actions,) action probability distribution
        """
        sh = _state_hash(state_vec)
        if sh not in self.d1_weights:
            return np.ones(self.n_actions) / self.n_actions

        go = self.d1_weights.get(sh, np.zeros(self.n_actions))
        nogo = self.d2_weights.get(sh, np.zeros(self.n_actions))
        net = go - nogo * 0.5

        # Softmax to probabilities
        net = np.clip(net, -5, 5)
        exp = np.exp(net - np.max(net))
        probs = exp / (exp.sum() + 1e-8)
        return probs.astype(np.float64)

    # ================================================================
    # Diagnostics
    # ================================================================

    def is_habitual(self, state_vec: np.ndarray) -> bool:
        """Check if behavior in given state has become habitual."""
        sh = _state_hash(state_vec)
        return self.habit_strength.get(sh, 0.0) > 0.7

    def devaluation_test(self, state_vec: np.ndarray, action: int,
                         da_signal_on_devaluation: float = -0.5) -> bool:
        """Devaluation diagnostic: test if behavior persists after outcome devaluation.

        This is the operational definition of habit — insensitivity to
        outcome value changes.

        Args:
            state_vec: sensory state
            action: action being tested
            da_signal_on_devaluation: DA signal after devaluation (negative = bad)

        Returns:
            True = habitual behavior (still chooses same action after devaluation)
            False = goal-directed (behavior decreases after devaluation)
        """
        sh = _state_hash(state_vec)
        hs = self.habit_strength.get(sh, 0.0)

        # Before devaluation: D1 strength for this action
        d1_before = self.d1_weights.get(
            sh, np.zeros(self.n_actions))[action]

        # Simulate devaluation: learn with negative DA signal
        self.learn(state_vec, action, da_signal_on_devaluation, reward=0.0)

        # After devaluation: habitual (high habit strength) → d1 barely changed
        #                   goal-directed (low habit strength) → d1 significantly decreased
        d1_after = self.d1_weights[sh][action]
        d1_change = abs(d1_after - d1_before)

        # Restore (devaluation test should not permanently alter weights)
        self.d1_weights[sh][action] = d1_before

        # Habit: d1 change is small (< 0.05)
        return d1_change < 0.05

    @property
    def global_habit_strength(self) -> float:
        """Global habitization level [0,1] — automation level of entire striatum."""
        return float(self._habit_ema)

    def get_state(self) -> dict:
        """Return striatum state summary."""
        return {
            'n_states_known': len(self.d1_weights),
            'n_learn_events': self.n_learn_events,
            'n_habitual_actions': self.n_habitual_actions,
            'global_habit_strength': self.global_habit_strength,
            'n_habitual_states': sum(
                1 for hs in self.habit_strength.values() if hs > 0.5),
            'd1_d2_balance': self.d1_d2_balance,
        }
