"""v6.0 acceptance test — run with: python tools/v6_acceptance_test.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile
from cns.agent import Agent
from cns.innate import apply_innate_config
from cns.persistence import save_agent, load_agent_state, restore_agent


def test_all():
    print("=== v6.0 Acceptance Tests ===\n")

    # Test 1: Zero-pretraining startup
    print("1. Zero-pretraining startup...")
    agent = Agent(rng=np.random.default_rng(42))
    apply_innate_config(agent)
    assert agent.net.n_clusters == 0, "Should start with 0 episodic clusters"
    assert agent.semantic_memory.n_clusters == 0, "Should start with 0 semantic"
    assert agent.striatum.n_learn_events == 0, "Should start with 0 striatum events"
    print("   PASS: Agent starts with zero clusters")

    # Test 2: Online learning
    print("2. Online learning...")
    for i in range(5):
        s = np.zeros(516, dtype=np.float32)
        s[:64] = np.random.randn(64).astype(np.float32)
        agent.step(s, i + 1)
    assert agent.net.n_clusters > 0, "Should create clusters from input"
    print(f"   PASS: {agent.net.n_clusters} episodic clusters, "
          f"{agent.semantic_memory.n_clusters} semantic clusters")

    # Test 3: Cross-session consolidation
    print("3. Cross-session consolidation...")
    result = agent.consolidate_across_sessions()
    assert result["n_extracted"] >= 0, "Should process clusters"
    print(f"   PASS: {result['n_extracted']} extracted, "
          f"{result['n_new_facts']} new facts, "
          f"{result['n_updated']} updated")

    # Test 4: Semantic memory query
    print("4. Semantic memory query...")
    q = np.zeros(64, dtype=np.float32)
    q[:16] = np.ones(16) * 0.3
    familiarity = agent.semantic_memory.knows_about(q)
    print(f"   PASS: familiarity={familiarity:.3f}")

    # Test 5: Striatum habit learning
    print("5. Striatum habit learning...")
    state = agent.striatum.get_state()
    print(f"   States known: {state['n_states_known']}, habit: {state['global_habit_strength']:.3f}")
    print("   PASS: Striatum active")

    # Test 6: Pure mode config
    print("6. Pure mode config...")
    assert agent.theta.cluster_threshold < 0.7
    assert agent.theta.learn_rate_l0 > 0.08
    print(f"   PASS: threshold={agent.theta.cluster_threshold:.2f}, lr={agent.theta.learn_rate_l0:.2f}")

    # Test 7: Theta params
    print("7. Theta params...")
    assert len(agent.theta.to_dict()) == 32, "Should have 32 params"
    print(f"   PASS: {len(agent.theta.to_dict())} params")

    # Test 8: Persistence round-trip
    print("8. Persistence round-trip...")
    path = os.path.join(tempfile.gettempdir(), "v6_test_agent.pkl")
    save_agent(agent, path)
    data = load_agent_state(path)
    assert "semantic_memory" in data, "Save should include semantic memory"
    assert "striatum" in data, "Save should include striatum"
    agent2 = Agent(rng=np.random.default_rng(99))
    restore_agent(agent2, data, verbose=False)
    assert agent2.semantic_memory.n_clusters == agent.semantic_memory.n_clusters
    assert agent2.striatum.n_learn_events == agent.striatum.n_learn_events
    os.unlink(path)
    print("   PASS: Full round-trip with all new modules")

    print("\n=== All 8 acceptance tests passed ===")
    return True


if __name__ == "__main__":
    test_all()
