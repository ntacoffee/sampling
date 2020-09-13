#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


class Gaussian:
    def __init__(self, mean, std, shape):
        self.mean = np.ones(shape) * mean
        self.std = std

    def prob_unnormalized(self, z):
        return np.exp(-(np.sum((z - self.mean) ** 2, axis=-1)) / (2.0 * self.std ** 2))


# 各種設定
L = 1000  # サンプル数
dim = 2  # 次元数
param_proposal = (2.0, 2.0)  # proposal distributionの平均と標準偏差
param_target = (1.0, 1.0)  # targetr distributionの平均と標準偏差
f = lambda x: np.sum(x ** 2 + 2 * x + 1, axis=-1)  # 期待値を取る対象の関数


def main():

    z_proposal = np.random.normal(*param_proposal, (L, dim))

    # 正規化されていない分布を定義
    q_tilde = Gaussian(*param_proposal, (L, dim)).prob_unnormalized
    p_tilde = Gaussian(*param_target, (L, dim)).prob_unnormalized

    # importance weight の計算
    r_tilde = p_tilde(z_proposal) / q_tilde(z_proposal)

    # 正規化定数の補正項の計算
    ZpZq = (1.0 / L) * np.sum(r_tilde)

    # importance sampling による平均値計算
    expectation_IS = (1.0 / L) * np.sum((r_tilde / ZpZq) * f(z_proposal))

    # 比較のため真の分布からの平均値計算
    z_target = np.random.normal(*param_target, (L, dim))
    expectation_true_sample = (1.0 / L) * np.sum(f(z_target))

    # 結果比較
    print(expectation_IS, expectation_true_sample)

    # どのサンプルで計算したか図示
    plt.scatter(z_proposal[:, 0], z_proposal[:, 1], label="proposal")
    plt.scatter(z_target[:, 0], z_target[:, 1], label="target")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
