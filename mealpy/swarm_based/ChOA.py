#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalChOA(Optimizer):
    """
    The original version of: Chimp Optimization Algorithm (ChOA)

    Links:
        https://doi.org/10.1016/j.eswa.2020.113338

    Notes:
        - 4 leaders (attacker, barrier, chaser, driver) = top-4 fitness
        - Leader roles are symbolic, mathematically identical
        - New position = mean of 4 leader-based moves

    References:
        Khishe, M., & Mosavi, M. R. (2020).
        Chimp optimization algorithm.
        Expert Systems with Applications, 149, 113338.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)

        
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])

        self.sort_flag = False
        self.is_parallelizable = False

    def evolve(self, epoch: int) -> None:
        """
        Main ChOA update equations
        """

        # Select best 4 agents (leaders)
        _, best_agents, _ = self.get_special_agents(
            self.pop, n_best=4, n_worst=0, minmax=self.problem.minmax
        )
        attacker, barrier, chaser, driver = best_agents

        # a = 2 - 2*(t/T)  controls exploration/exploitation
        a = 2.0 - 2.0 * epoch / self.epoch
        n_dims = self.problem.n_dims

        for idx in range(self.pop_size):
            X = self.pop[idx].solution  # current agent position

            # Leader-based position update
            def toward(leader_pos):
                r1 = self.generator.random(n_dims)     # random vector [0,1]
                r2 = self.generator.random(n_dims)

                A = 2.0 * a * r1 - a                   # Eq. (A): step size & direction
                C = 2.0 * r2                           # Eq. (C): leader influence
                D = np.abs(C * leader_pos - X)         # Eq. (D): distance to leader

                return leader_pos - A * D               # Eq. (Xk): new candidate

            # 4 candidate positions from 4 leaders
            X1 = toward(attacker.solution)
            X2 = toward(barrier.solution)
            X3 = toward(chaser.solution)
            X4 = toward(driver.solution)

            # X_new = (X1 + X2 + X3 + X4) / 4
            pos_new = (X1 + X2 + X3 + X4) / 4.0

            # sınırlar
            pos_new = self.correct_solution(pos_new)

            # Fitness 
            agent_new = self.generate_agent(pos_new)

            # Greedy selection 
            if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent_new
