import numpy as np
from mealpy.optimizer import Optimizer


class OriginalChOA(Optimizer):
    """
    ChOA (Chimp Optimization Algorithm) 
    Liderler: popülasyondaki fitness'a göre en iyi 4 agent.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch: int) -> None:
        # Lider seçimi: attacker/barrier/chaser/driver = best_1..best_4
        _, best_agents, _ = self.get_special_agents(self.pop, n_best=4, n_worst=0, minmax=self.problem.minmax)
        attacker, barrier, chaser, driver = best_agents

        # (Eq. a)  a = 2 - 2*(t/T)
        a = 2.0 - 2.0 * epoch / self.epoch
        n_dims = self.problem.n_dims

        for i in range(self.pop_size):
            X = self.pop[i].solution  # X_i(t)

            # Tek lider için ChOA adımı:
            # (Eq. A) A = 2*a*r1 - a
            # (Eq. C) C = 2*r2
            # (Eq. D) D = | C*X_leader - X |
            # (Eq. Xk) Xk = X_leader - A*D
            def toward(X_leader: np.ndarray) -> np.ndarray:
                r1 = self.generator.random(n_dims)
                r2 = self.generator.random(n_dims)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                D = np.abs(C * X_leader - X)
                return X_leader - A * D

            # (Eq. X1..X4) 4 liderden 4 aday konum
            X1 = toward(attacker.solution)
            X2 = toward(barrier.solution)
            X3 = toward(chaser.solution)
            X4 = toward(driver.solution)

            # (Eq. X_new) X_i(t+1) = (X1+X2+X3+X4)/4
            pos_new = (X1 + X2 + X3 + X4) / 4.0

            # Bounds: clip / düzeltme
            pos_new = self.correct_solution(pos_new)

            # Fitness değerlendirme: agent_new.target.fitness = f(pos_new)
            agent_new = self.generate_agent(pos_new)

            # Greedy kabul:
            # minimize ise f(new) < f(old) ise replace
            if self.compare_target(agent_new.target, self.pop[i].target, self.problem.minmax):
                self.pop[i] = agent_new
