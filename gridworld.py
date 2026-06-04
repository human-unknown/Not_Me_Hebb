"""
gridworld.py —— 网格世界环境
自由能原理智能体 — M1/M2/M3

10×10 网格:
- 1-3 个智能体从中心附近出发
- 5 个资源点：采集到获得 +1.0 奖励，资源重新随机生成
- 3 个障碍物：碰撞获得 -0.5 惩罚

感知向量 (D=48) 编码 (v2):
- [0:5]   : 身体标量 b
- [5:8]   : body 导数 + 碰撞
- [8:10]  : 归一化位置
- [10:20] : 5 个资源的距离 + 角度
- [20:26] : 3 个障碍物的距离 + 距离倒数
- [26:34] : 其他 agent 相对位置 (M3)
- [34:]   : 填充 0
"""

import numpy as np
from data_types import D, A, ACTION_DIRECTIONS


class GridWorld:
    """网格世界环境 — 支持单/多智能体

    提供物理世界模拟:
    - 智能体位置 (连续坐标，可在网格内自由移动)
    - 离散行动 (N,S,W,E)
    - 资源采集 + 障碍物惩罚
    - 感知向量构建（含社会感知 M3+）

    向后兼容: agent_pos/total_reward 属性委托到 agent 0。
    """

    def __init__(self, size: int = 10, n_agents: int = 1,
                 rng: np.random.Generator = None):
        self.size = float(size)
        self.n_agents = n_agents
        self.rng = rng if rng is not None else np.random.default_rng()

        # 多智能体位置：agent 0 在中心，其他从附近分散出发
        center = np.array([size / 2, size / 2], dtype=float)
        self.agent_positions: list[np.ndarray] = []
        for i in range(n_agents):
            if i == 0:
                # 向后兼容：单 agent 模式从中心出发
                self.agent_positions.append(center.copy())
            else:
                offset = np.array([(i % 2) * 1.8 - 0.9,
                                  (i // 2) * 1.8 - 0.9], dtype=float)
                self.agent_positions.append(
                    np.clip(center + offset, 0.5, size - 0.5))

        # 共享资源与障碍物
        self.resources = self._spawn_resources(5)
        self.obstacles = self._spawn_obstacles(3)
        self.resource_field = self._make_field(0.3)
        self.obstacle_field = self._make_field(0.1)
        self.terrain_field = self._make_field(0.5) * 2 - 1

        # 每个 agent 独立统计
        self.total_rewards = [0.0] * n_agents
        self.steps: int = 0

    # ---- 向后兼容属性 ----
    @property
    def agent_pos(self) -> np.ndarray:
        return self.agent_positions[0]

    @agent_pos.setter
    def agent_pos(self, value: np.ndarray):
        self.agent_positions[0] = value

    @property
    def total_reward(self) -> float:
        return self.total_rewards[0]

    @total_reward.setter
    def total_reward(self, value: float):
        self.total_rewards[0] = value

    # ---- 生成 ----
    def _spawn_resources(self, n: int) -> list[np.ndarray]:
        return [self.rng.uniform(0, self.size, 2) for _ in range(n)]

    def _make_field(self, density: float = 0.3) -> np.ndarray:
        """生成连续标量场 (v2)"""
        field = self.rng.random((int(self.size), int(self.size)))
        field = np.where(field < density, field / density, 0.0)
        return field

    def _spawn_obstacles(self, n: int) -> list[np.ndarray]:
        return [self.rng.integers(0, int(self.size), 2).astype(float)
                for _ in range(n)]

    # ---- 社会感知索引 ----
    def _other_agent_ids(self, agent_id: int) -> list[int]:
        """返回除 agent_id 外的其他 agent ID 列表"""
        return [i for i in range(self.n_agents) if i != agent_id]

    # ---- 辅助感知方法 ----
    def _gradient_at(self, pos, angle, field):
        dx, dy = np.cos(angle) * 0.5, np.sin(angle) * 0.5
        x1 = int(np.clip(pos[0]+dx, 0, self.size-1))
        y1 = int(np.clip(pos[1]+dy, 0, self.size-1))
        x0 = int(np.clip(pos[0], 0, self.size-1))
        y0 = int(np.clip(pos[1], 0, self.size-1))
        return float(field[x1, y1] - field[x0, y0])

    def _obstacle_dist(self, pos, angle):
        for step in range(1, int(self.size)):
            cx = int(np.clip(pos[0]+np.cos(angle)*step, 0, self.size-1))
            cy = int(np.clip(pos[1]+np.sin(angle)*step, 0, self.size-1))
            if self.obstacle_field[cx, cy] > 0.5:
                return 1.0 - step/self.size
        return 0.0

    # ---- 感知 ----
    def get_sensory(self, agent_id: int = 0,
                    body: np.ndarray = None) -> np.ndarray:
        """构建感知向量 s (D=138) — 128 dim sensory core (v3)

        body[0:16] vision[16:64] audio[64:80] social[80:112] time[112:128]
        [128:138] 留给行动-后果模式
        """
        import math
        s = np.zeros(D)
        my_pos = self.agent_positions[agent_id]
        x, y = int(my_pos[0]), int(my_pos[1])
        T_cycle = 1000.0

        # body [0:16]
        if body is not None:
            s[0:5] = body[:5]
        s[5] = my_pos[0] / self.size
        s[6] = my_pos[1] / self.size

        # vision [16:64]: 5x5 local field + gradient + depth
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                base = 16 + (dy+2)*5 + (dx+2)
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    s[base] = self.resource_field[nx, ny]
        for d in range(8):
            angle = d * np.pi / 4
            s[41+d] = self._gradient_at(my_pos, angle, self.resource_field)
            s[49+d] = self._obstacle_dist(my_pos, angle)

        # audio [64:80]: directional sound from other agents
        for j, op in enumerate(self.agent_positions):
            if j == agent_id: continue
            delta = op - my_pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6: continue
            direction = int((np.arctan2(delta[1], delta[0])/np.pi*2+2) % 4)
            energy = math.exp(-dist/3.0)
            s[64 + direction*2] += energy * 0.7
            s[65 + direction*2] += energy * 0.3
        s[72] = float(np.mean(self.resource_field))
        s[73] = float(np.std(self.resource_field))

        # social [80:112]: other agents' position/distance
        others = self._other_agent_ids(agent_id)
        for idx, oid in enumerate(others):
            base = 80 + idx * 8
            if base + 7 >= 128: break
            op = self.agent_positions[oid]
            delta = op - my_pos
            s[base:base+2] = delta / self.size
            s[base+6] = float(np.linalg.norm(delta)) / self.size

        # time [112:128]: phase + step counter
        s[112] = math.sin(2*math.pi*self.steps/T_cycle)
        s[113] = math.cos(2*math.pi*self.steps/T_cycle)
        s[114] = self.steps / max(self.steps, 1)

        return s  # [128:138] zero for action-consequence

    def get_visible_sensory(self, vision_radius: float = 3.0,
                            agent_id: int = 0) -> np.ndarray:
        """获取视野遮蔽的感知向量 (M2/M3)

        距离 > vision_radius 的资源感知被归零
        """
        s = self.get_sensory(agent_id)

        # v3: 遮蔽视野外的 vision field 格点
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx = int(self.agent_positions[agent_id][0]) + dx
                ny = int(self.agent_positions[agent_id][1]) + dy
                dist = np.linalg.norm([dx, dy])
                if dist > vision_radius:
                    base = 16 + (dy+2)*5 + (dx+2)
                    if base < 64:
                        s[base] = 0.0

        return s

    # ---- 行动 ----
    def step(self, action: int, agent_id: int = 0) -> float:
        """执行行动，返回该 agent 的奖励

        Args:
            action: 0=N, 1=S, 2=W, 3=E, 4=REST
            agent_id: 行动主体 ID

        Returns:
            reward
        """
        # REST 行动：不移动，不采集，纯恢复
        if action == 4:
            self.steps += 1
            return 0.0

        move = ACTION_DIRECTIONS[action]
        new_pos = self.agent_positions[agent_id] + move
        new_pos = np.clip(new_pos, 0, self.size - 1)

        reward = 0.0

        # 资源采集
        i = 0
        while i < len(self.resources):
            r = self.resources[i]
            if np.linalg.norm(new_pos - r) < 0.5:
                reward += 1.0
                self.resources[i] = self.rng.uniform(0, self.size, 2)
            i += 1

        # 障碍物碰撞
        for o in self.obstacles:
            if np.linalg.norm(new_pos - o) < 0.5:
                reward -= 0.5

        # 其他 agent 碰撞（M3+: 物理阻塞但不惩罚，社会动态由 F_social 驱动）
        for other_id in range(self.n_agents):
            if other_id == agent_id:
                continue
            if np.linalg.norm(new_pos - self.agent_positions[other_id]) < 0.3:
                pass  # 物理接触 → 不惩罚，由社会自由能提供信号

        # 更新
        self.agent_positions[agent_id] = new_pos
        self.total_rewards[agent_id] += reward
        self.steps += 1

        return reward

    # ---- 重置 ----
    def reset(self):
        """重置环境"""
        center = np.array([self.size / 2, self.size / 2], dtype=float)
        for i in range(self.n_agents):
            offset = np.array([(i % 2) * 1.5 - 0.75,
                              (i // 2) * 1.5 - 0.75], dtype=float)
            self.agent_positions[i] = np.clip(
                center + offset, 0.5, self.size - 0.5)
        self.resources = self._spawn_resources(5)
        self.obstacles = self._spawn_obstacles(3)
        self.resource_field = self._make_field(0.3)
        self.obstacle_field = self._make_field(0.1)
        self.terrain_field = self._make_field(0.5) * 2 - 1
        self.total_rewards = [0.0] * self.n_agents
        self.steps = 0

    # ---- 覆盖率 ----
    def compute_coverage(self, pos_history: list,
                         agent_id: int = 0) -> float:
        """计算网格覆盖率"""
        if not pos_history:
            return 0.0
        visited = set()
        grid_size = int(self.size)
        for pos in pos_history:
            cell = (int(np.clip(pos[0], 0, grid_size - 1)),
                    int(np.clip(pos[1], 0, grid_size - 1)))
            visited.add(cell)
        return len(visited) / (grid_size * grid_size)

    # ---- 摘要 ----
    def get_state_summary(self) -> dict:
        return {
            'agent_positions': [p.copy() for p in self.agent_positions],
            'n_resources': len(self.resources),
            'n_obstacles': len(self.obstacles),
            'total_rewards': list(self.total_rewards),
            'steps': self.steps,
        }
