import numpy as np
import gym
from gym import spaces

'''
Environment for M4O-BO (OPT-BO)
'''
class M4OEnv(gym.Env):
    def __init__(self,one_data):
        super(M4OEnv, self).__init__()
        # communication range
        self.trans_range = 300
        # transmission power
        self.trans_power = 0.2
        # loss exponent
        self.trans_loss = 4
        # background noise
        self.noise = 10 ** (-13.4)
        # computation price per bit
        self.computation_cost = 1e-7
        # wireless bandwidth
        self.bandWith = 1e7
        # computational intensity
        self.intensity = 1000
        # weight coefficient
        self.wc = 1
        # CPU cycles of cv0
        self.c0 = 8e8
        # CPU cycles of cv1
        self.c1 = 8e8
        # CPU cycles of cv2
        self.c2 = 8e8

        # state for training
        self.data_lib = one_data
        self.data = np.array(self.data_lib[np.random.randint(0, np.array(self.data_lib).shape[0])])
        self.state = self.data

        # computation of connectivity
        self.array_S_0 = np.zeros(9)
        self.array_S_1 = np.zeros(9)
        self.array_S_2 = np.zeros(9)
        self.array_S_3 = np.zeros(9)
        self.array_S_4 = np.zeros(9)
        self.array_S_5 = np.zeros(9)
        self.array_S_6 = np.zeros(9)
        self.array_S_7 = np.zeros(9)
        self.array_S_8 = np.zeros(9)
        self.array_S_9 = np.zeros(9)

        self.array_l_0 = np.zeros(9)
        self.array_l_1 = np.zeros(9)
        self.array_l_2 = np.zeros(9)
        self.array_l_3 = np.zeros(9)
        self.array_l_4 = np.zeros(9)
        self.array_l_5 = np.zeros(9)
        self.array_l_6 = np.zeros(9)
        self.array_l_7 = np.zeros(9)
        self.array_l_8 = np.zeros(9)
        self.array_l_9 = np.zeros(9)

        self.array_S = [self.array_S_0, self.array_S_1, self.array_S_2, self.array_S_3, self.array_S_4,
                        self.array_S_5, self.array_S_6, self.array_S_7, self.array_S_8, self.array_S_9]
        self.array_l = [self.array_l_0, self.array_l_1, self.array_l_2, self.array_l_3, self.array_l_4,
                        self.array_l_5, self.array_l_6, self.array_l_7, self.array_l_8, self.array_l_9]

        for j in range(17, 27):
            for i in range(j + 1, 27):
                if np.sqrt((self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 18] = 0
                    self.array_S[j - 17][i - 18] = 1000
                else:
                    self.array_S[j - 17][i - 18] = np.sqrt(
                        (self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                    self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[17 + 2 + 3 * (i - 17)] / self.state[17 + 2 + 3 * (j - 17)] < 0:
                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                        self.state[17 + 1 + 3 * (j - 17)] - self.state[17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) + self.state[17 + 3 * (j - 17)] - self.state[
                                                                17 + 3 * (i - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) -
                                                            self.state[17 + 3 * (i - 17)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

            for i in range(17, j):
                if np.sqrt((self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 17] = 0
                    self.array_S[j - 17][i - 17] = 1000
                else:
                    self.array_S[j - 17][i - 17] = np.sqrt(
                        (self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                    self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[47 + 2 + 3 * (i - 27)] / self.state[17 + 2 + 3 * (j - 17)] < 0:

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                        self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                        self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

        self.array_l_2_0 = np.zeros(7)
        self.array_l_2_1 = np.zeros(7)
        self.array_l_2_2 = np.zeros(7)

        arr_temp_0 = np.zeros(8)
        self.arr_index_0_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            for k in range(1, j):
                arr_temp_0[k - 1] = min(self.array_l[0][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_0[k - 2] = min(self.array_l[0][k - 1], self.array_l[k][j])

            self.array_l_2_0[j - 3] = max(arr_temp_0)
            self.arr_index_0_to_j[j - 3] = arr_temp_0.tolist().index(self.array_l_2_0[j - 3])

        for i in range(3, 10):
            if self.arr_index_0_to_j[i - 3] < i - 1:
                self.arr_index_0_to_j[i - 3] += 1
                continue
            if self.arr_index_0_to_j[i - 3] >= i - 1:
                self.arr_index_0_to_j[i - 3] += 2

        arr_temp_1 = np.zeros(8)
        self.arr_index_1_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_1[0] = min(self.array_l[1][0], self.array_l[0][j - 1])
            for k in range(2, j):
                arr_temp_1[k - 1] = min(self.array_l[1][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_1[k - 2] = min(self.array_l[1][k - 1], self.array_l[k][j])

            self.array_l_2_1[j - 3] = max(arr_temp_1)
            self.arr_index_1_to_j[j - 3] = arr_temp_1.tolist().index(self.array_l_2_1[j - 3])

        for i in range(3, 10):
            if 0 < self.arr_index_1_to_j[i - 3] < i - 1:
                self.arr_index_1_to_j[i - 3] += 1
                continue
            if self.arr_index_1_to_j[i - 3] >= i - 1:
                self.arr_index_1_to_j[i - 3] += 2

        arr_temp_2 = np.zeros(8)
        self.arr_index_2_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_2[0] = min(self.array_l[2][0], self.array_l[0][j - 1])
            arr_temp_2[1] = min(self.array_l[2][1], self.array_l[1][j - 1])
            for k in range(3, j):
                arr_temp_2[k - 1] = min(self.array_l[2][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_2[k - 2] = min(self.array_l[2][k - 1], self.array_l[k][j])

            self.array_l_2_2[j - 3] = max(arr_temp_2)
            self.arr_index_2_to_j[j - 3] = arr_temp_2.tolist().index(self.array_l_2_2[j - 3])

        for i in range(3, 10):
            if 1 < self.arr_index_2_to_j[i - 3] < i - 1:
                self.arr_index_2_to_j[i - 3] += 1
                continue
            if self.arr_index_2_to_j[i - 3] >= i - 1:
                self.arr_index_2_to_j[i - 3] += 2

        self.Array_l_0 = np.zeros(7)
        self.Array_l_1 = np.zeros(7)
        self.Array_l_2 = np.zeros(7)
        for _ in range(7):
            self.Array_l_0[_] = max(self.array_l[0][_ + 2], self.array_l_2_0[_])
        for _ in range(7):
            self.Array_l_1[_] = max(self.array_l[1][_ + 2], self.array_l_2_1[_])
        for _ in range(7):
            self.Array_l_2[_] = max(self.array_l[2][_ + 2], self.array_l_2_2[_])

        # action space
        # action for SV selection
        self.action_space = spaces.MultiDiscrete([8]*3)

        # state space
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(47,), dtype=np.float32)

        self.step_count = 0
        # number of tasks per CV
        self.max_steps = 5
        self.max_time = 0

    # reset when an apisode ends
    def reset(self):
        self.step_count = 0
        i = np.random.randint(0,np.array(self.data_lib).shape[0])
        self.data = np.array(self.data_lib[i])
        self.state = self.data
        obs = self.state
        return obs

    # step to next timestep in an episode
    def step(self, action):
        self.step_count += 1
        P_1, P_2, P_3 = action

        cost, delay = self._compute_cost(P_1, P_2, P_3)
        self.max_time = delay
        reward = -cost

        done = self.step_count == self.max_steps

        # update positions
        for i in range(17, len(self.state), 3):
            x, y, z = self.state[i], self.state[i + 1], self.state[i + 2]
            self.state[i] = x + self.max_time * z

        # update connectivity
        for j in range(17, 27):
            for i in range(j + 1, 27):
                if np.sqrt((self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 18] = 0
                    self.array_S[j - 17][i - 18] = 1000
                else:
                    self.array_S[j - 17][i - 18] = np.sqrt(
                        (self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[17 + 2 + 3 * (i - 17)] / self.state[17 + 2 + 3 * (j - 17)] < 0:
                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) + self.state[17 + 3 * (j - 17)] - self.state[
                                                                17 + 3 * (i - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) -
                                                            self.state[17 + 3 * (i - 17)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

            for i in range(17, j):
                if np.sqrt((self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 17] = 0
                    self.array_S[j - 17][i - 17] = 1000
                else:
                    self.array_S[j - 17][i - 17] = np.sqrt(
                        (self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[47 + 2 + 3 * (i - 27)] / self.state[17 + 2 + 3 * (j - 17)] < 0:

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

        self.array_l_2_0 = np.zeros(7)
        self.array_l_2_1 = np.zeros(7)
        self.array_l_2_2 = np.zeros(7)

        arr_temp_0 = np.zeros(8)
        self.arr_index_0_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            for k in range(1, j):
                arr_temp_0[k - 1] = min(self.array_l[0][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_0[k - 2] = min(self.array_l[0][k - 1], self.array_l[k][j])

            self.array_l_2_0[j - 3] = max(arr_temp_0)
            self.arr_index_0_to_j[j - 3] = arr_temp_0.tolist().index(self.array_l_2_0[j - 3])

        for i in range(3, 10):
            if self.arr_index_0_to_j[i - 3] < i - 1:
                self.arr_index_0_to_j[i - 3] += 1
                continue
            if self.arr_index_0_to_j[i - 3] >= i - 1:
                self.arr_index_0_to_j[i - 3] += 2

        arr_temp_1 = np.zeros(8)
        self.arr_index_1_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_1[0] = min(self.array_l[1][0], self.array_l[0][j - 1])
            for k in range(2, j):
                arr_temp_1[k - 1] = min(self.array_l[1][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_1[k - 2] = min(self.array_l[1][k - 1], self.array_l[k][j])

            self.array_l_2_1[j - 3] = max(arr_temp_1)
            self.arr_index_1_to_j[j - 3] = arr_temp_1.tolist().index(self.array_l_2_1[j - 3])

        for i in range(3, 10):
            if 0 < self.arr_index_1_to_j[i - 3] < i - 1:
                self.arr_index_1_to_j[i - 3] += 1
                continue
            if self.arr_index_1_to_j[i - 3] >= i - 1:
                self.arr_index_1_to_j[i - 3] += 2

        arr_temp_2 = np.zeros(8)
        self.arr_index_2_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_2[0] = min(self.array_l[2][0], self.array_l[0][j - 1])
            arr_temp_2[1] = min(self.array_l[2][1], self.array_l[1][j - 1])
            for k in range(3, j):
                arr_temp_2[k - 1] = min(self.array_l[2][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_2[k - 2] = min(self.array_l[2][k - 1], self.array_l[k][j])

            self.array_l_2_2[j - 3] = max(arr_temp_2)
            self.arr_index_2_to_j[j - 3] = arr_temp_2.tolist().index(self.array_l_2_2[j - 3])

        for i in range(3, 10):
            if 1 < self.arr_index_2_to_j[i - 3] < i - 1:
                self.arr_index_2_to_j[i - 3] += 1
                continue
            if self.arr_index_2_to_j[i - 3] >= i - 1:
                self.arr_index_2_to_j[i - 3] += 2

        self.Array_l_0 = np.zeros(7)
        self.Array_l_1 = np.zeros(7)
        self.Array_l_2 = np.zeros(7)
        for _ in range(7):
            self.Array_l_0[_] = max(self.array_l[0][_ + 2], self.array_l_2_0[_])
        for _ in range(7):
            self.Array_l_1[_] = max(self.array_l[1][_ + 2], self.array_l_2_1[_])
        for _ in range(7):
            self.Array_l_2[_] = max(self.array_l[2][_ + 2], self.array_l_2_2[_])

        obs = self.state

        return obs, float(reward), done, {}

    # computation of total cost of each timestep
    def _compute_cost(self, P_1, P_2, P_3):

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, c3, c4, c5, c6, c7, c8, c9, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = self.state

        # task in each vehicle
        c0_task = 0
        c1_task = 0
        c2_task = 0

        c3_task_0 = 0
        c3_task_1 = 0
        c3_task_2 = 0

        c4_task_0 = 0
        c4_task_1 = 0
        c4_task_2 = 0

        c5_task_0 = 0
        c5_task_1 = 0
        c5_task_2 = 0

        c6_task_0 = 0
        c6_task_1 = 0
        c6_task_2 = 0

        c7_task_0 = 0
        c7_task_1 = 0
        c7_task_2 = 0

        c8_task_0 = 0
        c8_task_1 = 0
        c8_task_2 = 0

        c9_task_0 = 0
        c9_task_1 = 0
        c9_task_2 = 0

        # transmission delay
        c3_trans_0 = 0
        c3_trans_1 = 0
        c3_trans_2 = 0

        c4_trans_0 = 0
        c4_trans_1 = 0
        c4_trans_2 = 0

        c5_trans_0 = 0
        c5_trans_1 = 0
        c5_trans_2 = 0

        c6_trans_0 = 0
        c6_trans_1 = 0
        c6_trans_2 = 0

        c7_trans_0 = 0
        c7_trans_1 = 0
        c7_trans_2 = 0

        c8_trans_0 = 0
        c8_trans_1 = 0
        c8_trans_2 = 0

        c9_trans_0 = 0
        c9_trans_1 = 0
        c9_trans_2 = 0

        # execution delay
        c3_exe_0 = 0
        c3_exe_1 = 0
        c3_exe_2 = 0

        c4_exe_0 = 0
        c4_exe_1 = 0
        c4_exe_2 = 0

        c5_exe_0 = 0
        c5_exe_1 = 0
        c5_exe_2 = 0

        c6_exe_0 = 0
        c6_exe_1 = 0
        c6_exe_2 = 0

        c7_exe_0 = 0
        c7_exe_1 = 0
        c7_exe_2 = 0

        c8_exe_0 = 0
        c8_exe_1 = 0
        c8_exe_2 = 0

        c9_exe_0 = 0
        c9_exe_1 = 0
        c9_exe_2 = 0

        # delay of each task in each SV
        c3_delay_0 = 0
        c3_delay_1 = 0
        c3_delay_2 = 0

        c4_delay_0 = 0
        c4_delay_1 = 0
        c4_delay_2 = 0

        c5_delay_0 = 0
        c5_delay_1 = 0
        c5_delay_2 = 0

        c6_delay_0 = 0
        c6_delay_1 = 0
        c6_delay_2 = 0

        c7_delay_0 = 0
        c7_delay_1 = 0
        c7_delay_2 = 0

        c8_delay_0 = 0
        c8_delay_1 = 0
        c8_delay_2 = 0

        c9_delay_0 = 0
        c9_delay_1 = 0
        c9_delay_2 = 0

        # initial punish
        punish = 0

        # resource allocation for each SV
        c_map = {
            1: c3,
            2: c4,
            3: c5,
            4: c6,
            5: c7,
            6: c8,
            7: c9
        }
        for val in range(1, 8):
            count = [P_1, P_2, P_3].count(val)
            if count == 2:
                c_map[val] /= 2
            elif count == 3:
                c_map[val] /= 3
        c3, c4, c5, c6, c7, c8, c9 = c_map[1], c_map[2], c_map[3], c_map[4], c_map[5], c_map[6], c_map[7]

        if P_1 == 0:
            c0_task += t1
        if P_2 == 0:
            c1_task += t2
        if P_3 == 0:
            c2_task += t3

        # cost in SV3
        if P_1 == 1:
            if self.Array_l_0[0] == 0:
                punish += 100
            else:
                c3_task_0 += t1
                if self.array_l[0][2] >= self.array_l_2_0[0]:
                    c3_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][2] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_0_to_j[0] < 3:
                        c3_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        0] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[0]][2] ** self.trans_loss) * self.noise)))
                    else:
                        c3_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        0] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[0]][3] ** self.trans_loss) * self.noise)))

                c3_exe_0 += c3_task_0 / c3
                c3_delay_0 += c3_trans_0 + c3_exe_0
                if c3_delay_0 > self.Array_l_0[0]:
                    punish += 100

        if P_2 == 1:
            if self.Array_l_1[0] == 0:
                punish += 100
            else:
                c3_task_1 += t2
                if self.array_l[1][2] >= self.array_l_2_1[0]:
                    c3_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][2] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[0] < 3:
                        c3_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        0] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[0]][2] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[0] == 0:
                        c3_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[0]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[0]][2] ** self.trans_loss) * self.noise)))
                    else:
                        c3_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        0] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[0]][3] ** self.trans_loss) * self.noise)))

                c3_exe_1 += c3_task_1 / c3
                c3_delay_1 += c3_trans_1 + c3_exe_1
                if c3_delay_1 > self.Array_l_1[0]:
                    punish += 100


        if P_3 == 1:
            if self.Array_l_2[0] == 0:
                punish += 100
            else:
                c3_task_2 += t3
                if self.array_l[2][2] >= self.array_l_2_2[0]:
                    c3_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][2] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_2_to_j[0] < 3:
                        c3_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[0]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[0]][2] ** self.trans_loss) * self.noise)))
                    else:
                        c3_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        0] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[0]][3] ** self.trans_loss) * self.noise)))

                c3_exe_2 += c3_task_2 / c3
                c3_delay_2 += c3_trans_2 + c3_exe_2
                if c3_delay_2 > self.Array_l_2[0]:
                    punish += 100

        c3_delay = max(c3_delay_0, c3_delay_1, c3_delay_2)
        c3_task = c3_task_0 + c3_task_1 + c3_task_2
        c3_cost = c3_task / self.intensity * self.computation_cost

        # cost in SV4
        if P_1 == 2:
            if self.Array_l_0[1] == 0:
                punish += 100
            else:
                c4_task_0 += t1
                if self.array_l[0][3] >= self.array_l_2_0[1]:
                    c4_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][3] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_0_to_j[1] < 4:
                        c4_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        1] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[1]][3] ** self.trans_loss) * self.noise)))
                    else:
                        c4_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        1] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[1]][4] ** self.trans_loss) * self.noise)))

                c4_exe_0 += c4_task_0 / c4
                c4_delay_0 += c4_trans_0 + c4_exe_0
                if c4_delay_0 > self.Array_l_0[1]:
                    punish += 100

        if P_2 == 2:
            if self.Array_l_1[1] == 0:
                punish += 100
            else:
                c4_task_1 += t2
                if self.array_l[1][3] >= self.array_l_2_1[1]:
                    c4_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][3] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[1] < 4:
                        c4_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        1] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[1]][3] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[1] == 0:
                        c4_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[1]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[1]][3] ** self.trans_loss) * self.noise)))
                    else:
                        c4_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        1] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[1]][4] ** self.trans_loss) * self.noise)))

                c4_exe_1 += c4_task_1 / c4
                c4_delay_1 += c4_trans_1 + c4_exe_1
                if c4_delay_1 > self.Array_l_1[1]:
                    punish += 100

        if P_3 == 2:
            if self.Array_l_2[1] == 0:
                punish += 100
            else:
                c4_task_2 += t3
                if self.array_l[2][3] >= self.array_l_2_2[1]:
                    c4_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][3] ** self.trans_loss) * self.noise)))
                else:
                    if 2 < self.arr_index_2_to_j[1] < 4:
                        c4_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        1] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[1]][3] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_2_to_j[1] < 2:
                        c4_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[1]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[1]][3] ** self.trans_loss) * self.noise)))
                    else:
                        c4_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        1] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[1]][4] ** self.trans_loss) * self.noise)))

                c4_exe_2 += c4_task_2 / c4
                c4_delay_2 += c4_trans_2 + c4_exe_2
                if c4_delay_2 > self.Array_l_2[1]:
                    punish += 100

        c4_delay = max(c4_delay_0, c4_delay_1, c4_delay_2)
        c4_task = c4_task_0 + c4_task_1 + c4_task_2
        c4_cost = c4_task / self.intensity * self.computation_cost

        # cost in SV5
        if P_1 == 3:
            if self.Array_l_0[2] == 0:
                punish += 100
            else:
                c5_task_0 += t1
                if self.array_l[0][4] >= self.array_l_2_0[2]:
                    c5_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][4] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_0_to_j[2] < 5:
                        c5_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        2] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[2]][4] ** self.trans_loss) * self.noise)))
                    else:
                        c5_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        2] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[2]][5] ** self.trans_loss) * self.noise)))

                c5_exe_0 += c5_task_0 / c5
                c5_delay_0 += c5_trans_0 + c5_exe_0
                if c5_delay_0 > self.Array_l_0[2]:
                    punish += 100

        if P_2 == 3:
            if self.Array_l_1[2] == 0:
                punish += 100
            else:
                c5_task_1 += t2
                if self.array_l[1][4] >= self.array_l_2_1[2]:
                    c5_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][4] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[2] < 5:
                        c5_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        2] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[2]][4] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[2] == 0:
                        c5_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[2]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[2]][4] ** self.trans_loss) * self.noise)))
                    else:
                        c5_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        2] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[2]][5] ** self.trans_loss) * self.noise)))

                c5_exe_1 += c5_task_1 / c5
                c5_delay_1 += c5_trans_1 + c5_exe_1
                if c5_delay_1 > self.Array_l_1[2]:
                    punish += 100

        if P_3 == 3:
            if self.Array_l_2[2] == 0:
                punish += 100
            else:
                c5_task_2 += t3
                if self.array_l[2][4] >= self.array_l_2_2[2]:
                    c5_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][4] ** self.trans_loss) * self.noise)))
                else:
                    if 2 < self.arr_index_2_to_j[2] < 5:
                        c5_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        2] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[2]][4] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_2_to_j[2] < 2:
                        c5_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[2]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[2]][4] ** self.trans_loss) * self.noise)))
                    else:
                        c5_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        2] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[2]][5] ** self.trans_loss) * self.noise)))

                c5_exe_2 += c5_task_2 / c5
                c5_delay_2 += c5_trans_2 + c5_exe_2
                if c5_delay_2 > self.Array_l_2[2]:
                    punish += 100

        c5_delay = max(c5_delay_0, c5_delay_1, c5_delay_2)
        c5_task = c5_task_0 + c5_task_1 + c5_task_2
        c5_cost = c5_task / self.intensity * self.computation_cost

        # cost in SV6
        if P_1 == 4:
            if self.Array_l_0[3] == 0:
                punish += 100
            else:
                c6_task_0 += t1
                if self.array_l[0][5] >= self.array_l_2_0[3]:
                    c6_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][5] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_0_to_j[3] < 6:
                        c6_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        3] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[3]][5] ** self.trans_loss) * self.noise)))
                    else:
                        c6_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        3] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[3]][6] ** self.trans_loss) * self.noise)))

                c6_exe_0 += c6_task_0 / c6
                c6_delay_0 += c6_trans_0 + c6_exe_0
                if c6_delay_0 > self.Array_l_0[3]:
                    punish += 100

        if P_2 == 4:
            if self.Array_l_1[3] == 0:
                punish += 100
            else:
                c6_task_1 += t2
                if self.array_l[1][5] >= self.array_l_2_1[3]:
                    c6_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][5] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[3] < 6:
                        c6_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        3] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[3]][5] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[3] == 0:
                        c6_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[3]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[3]][5] ** self.trans_loss) * self.noise)))
                    else:
                        c6_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        3] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[3]][6] ** self.trans_loss) * self.noise)))

                c6_exe_1 += c6_task_1 / c6
                c6_delay_1 += c6_trans_1 + c6_exe_1
                if c6_delay_1 > self.Array_l_1[3]:
                    punish += 100

        if P_3 == 4:
            if self.Array_l_2[3] == 0:
                punish += 100
            else:
                c6_task_2 += t3
                if self.array_l[2][5] >= self.array_l_2_2[3]:
                    c6_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][5] ** self.trans_loss) * self.noise)))
                else:
                    if 2 < self.arr_index_2_to_j[3] < 6:
                        c6_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        3] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[3]][5] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_2_to_j[3] < 2:
                        c6_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[3]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[3]][5] ** self.trans_loss) * self.noise)))
                    else:
                        c6_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        3] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[3]][6] ** self.trans_loss) * self.noise)))

                c6_exe_2 += c6_task_2 / c6
                c6_delay_2 += c6_trans_2 + c6_exe_2
                if c6_delay_2 > self.Array_l_2[3]:
                    punish += 100

        c6_delay = max(c6_delay_0, c6_delay_1, c6_delay_2)
        c6_task = c6_task_0 + c6_task_1 + c6_task_2
        c6_cost = c6_task / self.intensity * self.computation_cost

        # cost in SV7
        if P_1 == 5:
            if self.Array_l_0[4] == 0:
                punish += 100
            else:
                c7_task_0 += t1
                if self.array_l[0][6] >= self.array_l_2_0[4]:
                    c7_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][6] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_0_to_j[4] < 7:
                        c7_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        4] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[4]][6] ** self.trans_loss) * self.noise)))
                    else:
                        c7_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        4] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[4]][7] ** self.trans_loss) * self.noise)))

                c7_exe_0 += c7_task_0 / c7
                c7_delay_0 += c7_trans_0 + c7_exe_0
                if c7_delay_0 > self.Array_l_0[4]:
                    punish += 100

        if P_2 == 5:
            if self.Array_l_1[4] == 0:
                punish += 100
            else:
                c7_task_1 += t2
                if self.array_l[1][6] >= self.array_l_2_1[4]:
                    c7_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][6] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[4] < 7:
                        c7_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        4] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[4]][6] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[4] == 0:
                        c7_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[4]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[4]][6] ** self.trans_loss) * self.noise)))
                    else:
                        c7_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        4] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[4]][7] ** self.trans_loss) * self.noise)))

                c7_exe_1 += c7_task_1 / c7
                c7_delay_1 += c7_trans_1 + c7_exe_1
                if c7_delay_1 > self.Array_l_1[4]:
                    punish += 100

        if P_3 == 5:
            if self.Array_l_2[4] == 0:
                punish += 100
            else:
                c7_task_2 += t3
                if self.array_l[2][6] >= self.array_l_2_2[4]:
                    c7_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][6] ** self.trans_loss) * self.noise)))
                else:
                    if 2 < self.arr_index_2_to_j[4] < 7:
                        c7_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        4] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[4]][6] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_2_to_j[4] < 2:
                        c7_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[4]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[4]][6] ** self.trans_loss) * self.noise)))
                    else:
                        c7_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        4] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[4]][7] ** self.trans_loss) * self.noise)))

                c7_exe_2 += c7_task_2 / c7
                c7_delay_2 += c7_trans_2 + c7_exe_2
                if c7_delay_2 > self.Array_l_2[4]:
                    punish += 100

        c7_delay = max(c7_delay_0, c7_delay_1, c7_delay_2)
        c7_task = c7_task_0 + c7_task_1 + c7_task_2
        c7_cost = c7_task / self.intensity * self.computation_cost

        # cost in SV8
        if P_1 == 6:
            if self.Array_l_0[5] == 0:
                punish += 100
            else:
                c8_task_0 += t1
                if self.array_l[0][7] >= self.array_l_2_0[5]:
                    c8_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][7] ** self.trans_loss) * self.noise)))
                else:
                    if self.arr_index_0_to_j[5] < 8:
                        c8_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        5] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[5]][7] ** self.trans_loss) * self.noise)))
                    else:
                        c8_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[0][self.arr_index_0_to_j[
                                                                        5] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t1 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_0_to_j[5]][8] ** self.trans_loss) * self.noise)))

                c8_exe_0 += c8_task_0 / c8
                c8_delay_0 += c8_trans_0 + c8_exe_0
                if c8_delay_0 > self.Array_l_0[5]:
                    punish += 100

        if P_2 == 6:
            if self.Array_l_1[5] == 0:
                punish += 100
            else:
                c8_task_1 += t2
                if self.array_l[1][7] >= self.array_l_2_1[5]:
                    c8_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][7] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[5] < 8:
                        c8_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        5] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[5]][7] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[5] == 0:
                        c8_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[5]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[5]][7] ** self.trans_loss) * self.noise)))
                    else:
                        c8_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        5] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[5]][8] ** self.trans_loss) * self.noise)))

                c8_exe_1 += c8_task_1 / c8
                c8_delay_1 += c8_trans_1 + c8_exe_1
                if c8_delay_1 > self.Array_l_1[5]:
                    punish += 100

        if P_3 == 6:
            if self.Array_l_2[5] == 0:
                punish += 100
            else:
                c8_task_2 += t3
                if self.array_l[2][7] >= self.array_l_2_2[5]:
                    c8_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][7] ** self.trans_loss) * self.noise)))
                else:
                    if 2 < self.arr_index_2_to_j[5] < 8:
                        c8_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        5] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[5]][7] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_2_to_j[5] < 2:
                        c8_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[5]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[5]][7] ** self.trans_loss) * self.noise)))
                    else:
                        c8_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        5] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[5]][8] ** self.trans_loss) * self.noise)))

                c8_exe_2 += c8_task_2 / c8
                c8_delay_2 += c8_trans_2 + c8_exe_2
                if c8_delay_2 > self.Array_l_2[5]:
                    punish += 100

        c8_delay = max(c8_delay_0, c8_delay_1, c8_delay_2)
        c8_task = c8_task_0 + c8_task_1 + c8_task_2
        c8_cost = c8_task / self.intensity * self.computation_cost

        # cost in SV9
        if P_1 == 7:
            if self.Array_l_0[6] == 0:
                punish += 100
            else:
                c9_task_0 += t1
                if self.array_l[0][8] >= self.array_l_2_0[6]:
                    c9_trans_0 += (t1 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[0][8] ** self.trans_loss) * self.noise)))
                else:
                    c9_trans_0 += (t1 / self.intensity) / (self.bandWith * np.log2(
                        1 + self.trans_power / (
                                    (self.array_S[0][self.arr_index_0_to_j[6] - 1] ** self.trans_loss) * self.noise))) + \
                                  (t1 / self.intensity) / (self.bandWith * np.log2(
                        1 + self.trans_power / (
                                    (self.array_S[self.arr_index_0_to_j[6]][8] ** self.trans_loss) * self.noise)))

                c9_exe_0 += c9_task_0 / c9
                c9_delay_0 += c9_trans_0 + c9_exe_0
                if c9_delay_0 > self.Array_l_0[6]:
                    punish += 100

        if P_2 == 7:
            if self.Array_l_1[6] == 0:
                punish += 100
            else:
                c9_task_1 += t2
                if self.array_l[1][7] >= self.array_l_2_1[5]:
                    c9_trans_1 += (t2 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[1][7] ** self.trans_loss) * self.noise)))
                else:
                    if 0 < self.arr_index_1_to_j[6] < 9:
                        c9_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[1][self.arr_index_1_to_j[
                                                                        6] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[6]][8] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_1_to_j[6] == 0:
                        c9_trans_1 += (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[1][self.arr_index_1_to_j[6]] ** self.trans_loss) * self.noise))) + \
                                      (t2 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_1_to_j[6]][8] ** self.trans_loss) * self.noise)))

                c9_exe_1 += c9_task_1 / c9
                c9_delay_1 += c9_trans_1 + c9_exe_1
                if c9_delay_1 > self.Array_l_1[6]:
                    punish += 100

        if P_3 == 7:
            if self.Array_l_2[6] == 0:
                punish += 100
            else:
                c9_task_2 += t3
                if self.array_l[2][8] >= self.array_l_2_2[6]:
                    c9_trans_2 += (t3 / self.intensity) / (
                            self.bandWith * np.log2(
                        1 + self.trans_power / ((self.array_S[2][8] ** self.trans_loss) * self.noise)))
                else:
                    if 2 < self.arr_index_2_to_j[6] < 9:
                        c9_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / ((self.array_S[2][self.arr_index_2_to_j[
                                                                        6] - 1] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[6]][8] ** self.trans_loss) * self.noise)))
                    elif self.arr_index_2_to_j[6] < 2:
                        c9_trans_2 += (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[2][self.arr_index_2_to_j[6]] ** self.trans_loss) * self.noise))) + \
                                      (t3 / self.intensity) / (self.bandWith * np.log2(
                            1 + self.trans_power / (
                                        (self.array_S[self.arr_index_2_to_j[6]][8] ** self.trans_loss) * self.noise)))

                c9_exe_2 += c9_task_2 / c9
                c9_delay_2 += c9_trans_2 + c9_exe_2
                if c9_delay_2 > self.Array_l_2[6]:
                    punish += 100

        c9_delay = max(c9_delay_0, c9_delay_1, c9_delay_2)
        c9_task = c9_task_0 + c9_task_1 + c9_task_2
        c9_cost = c9_task / self.intensity * self.computation_cost

        c0_delay = c0_task / self.c0
        c1_delay = c1_task / self.c1
        c2_delay = c2_task / self.c2

        # tatal delay of each timestep
        delay = max(c0_delay, c1_delay, c2_delay, c3_delay, c4_delay, c5_delay, c6_delay, c7_delay, c8_delay, c9_delay)

        # tatal cost of each timestep
        cost = delay + self.wc*(c9_cost + c8_cost + c7_cost + c6_cost + c5_cost + c4_cost + c3_cost) + punish

        return cost, delay



'''
Environment for SH-BO
'''

'''
class M4OEnv(gym.Env):
    def __init__(self,one_data):
        super(M4OEnv, self).__init__()
        self.trans_range = 300
        self.trans_power = 0.2
        self.trans_loss = 4
        self.noise = 10 ** (-13.4)
        self.computation_cost = 1e-7
        self.bandWith = 1e7
        self.intensity = 1000
        self.wc = 1
        self.c0 = 8e8
        self.c1 = 8e8
        self.c2 = 8e8

        self.data_lib = one_data
        self.data = np.array(self.data_lib[np.random.randint(0, np.array(self.data_lib).shape[0])])
        self.state = self.data

        self.array_S_0 = np.zeros(9)
        self.array_S_1 = np.zeros(9)
        self.array_S_2 = np.zeros(9)
        self.array_S_3 = np.zeros(9)
        self.array_S_4 = np.zeros(9)
        self.array_S_5 = np.zeros(9)
        self.array_S_6 = np.zeros(9)
        self.array_S_7 = np.zeros(9)
        self.array_S_8 = np.zeros(9)
        self.array_S_9 = np.zeros(9)

        self.array_l_0 = np.zeros(9)
        self.array_l_1 = np.zeros(9)
        self.array_l_2 = np.zeros(9)
        self.array_l_3 = np.zeros(9)
        self.array_l_4 = np.zeros(9)
        self.array_l_5 = np.zeros(9)
        self.array_l_6 = np.zeros(9)
        self.array_l_7 = np.zeros(9)
        self.array_l_8 = np.zeros(9)
        self.array_l_9 = np.zeros(9)

        self.array_S = [self.array_S_0, self.array_S_1, self.array_S_2, self.array_S_3, self.array_S_4,
                        self.array_S_5, self.array_S_6, self.array_S_7, self.array_S_8, self.array_S_9]
        self.array_l = [self.array_l_0, self.array_l_1, self.array_l_2, self.array_l_3, self.array_l_4,
                        self.array_l_5, self.array_l_6, self.array_l_7, self.array_l_8, self.array_l_9]

        for j in range(17, 27):
            for i in range(j + 1, 27):
                if np.sqrt((self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 18] = 0
                    self.array_S[j - 17][i - 18] = 1000
                else:
                    self.array_S[j - 17][i - 18] = np.sqrt(
                        (self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                    self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[17 + 2 + 3 * (i - 17)] / self.state[17 + 2 + 3 * (j - 17)] < 0:
                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                        self.state[17 + 1 + 3 * (j - 17)] - self.state[17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) + self.state[17 + 3 * (j - 17)] - self.state[
                                                                17 + 3 * (i - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) -
                                                            self.state[17 + 3 * (i - 17)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

            for i in range(17, j):
                if np.sqrt((self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 17] = 0
                    self.array_S[j - 17][i - 17] = 1000
                else:
                    self.array_S[j - 17][i - 17] = np.sqrt(
                        (self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                    self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[47 + 2 + 3 * (i - 27)] / self.state[17 + 2 + 3 * (j - 17)] < 0:

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                        self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                        self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                        self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

        self.array_l_2_0 = np.zeros(7)
        self.array_l_2_1 = np.zeros(7)
        self.array_l_2_2 = np.zeros(7)

        arr_temp_0 = np.zeros(8)
        self.arr_index_0_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            for k in range(1, j):
                arr_temp_0[k - 1] = min(self.array_l[0][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_0[k - 2] = min(self.array_l[0][k - 1], self.array_l[k][j])

            self.array_l_2_0[j - 3] = max(arr_temp_0)
            self.arr_index_0_to_j[j - 3] = arr_temp_0.tolist().index(self.array_l_2_0[j - 3])

        for i in range(3, 10):
            if self.arr_index_0_to_j[i - 3] < i - 1:
                self.arr_index_0_to_j[i - 3] += 1
                continue
            if self.arr_index_0_to_j[i - 3] >= i - 1:
                self.arr_index_0_to_j[i - 3] += 2

        arr_temp_1 = np.zeros(8)
        self.arr_index_1_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_1[0] = min(self.array_l[1][0], self.array_l[0][j - 1])
            for k in range(2, j):
                arr_temp_1[k - 1] = min(self.array_l[1][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_1[k - 2] = min(self.array_l[1][k - 1], self.array_l[k][j])

            self.array_l_2_1[j - 3] = max(arr_temp_1)
            self.arr_index_1_to_j[j - 3] = arr_temp_1.tolist().index(self.array_l_2_1[j - 3])

        for i in range(3, 10):
            if 0 < self.arr_index_1_to_j[i - 3] < i - 1:
                self.arr_index_1_to_j[i - 3] += 1
                continue
            if self.arr_index_1_to_j[i - 3] >= i - 1:
                self.arr_index_1_to_j[i - 3] += 2

        arr_temp_2 = np.zeros(8)
        self.arr_index_2_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_2[0] = min(self.array_l[2][0], self.array_l[0][j - 1])
            arr_temp_2[1] = min(self.array_l[2][1], self.array_l[1][j - 1])
            for k in range(3, j):
                arr_temp_2[k - 1] = min(self.array_l[2][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_2[k - 2] = min(self.array_l[2][k - 1], self.array_l[k][j])

            self.array_l_2_2[j - 3] = max(arr_temp_2)
            self.arr_index_2_to_j[j - 3] = arr_temp_2.tolist().index(self.array_l_2_2[j - 3])

        for i in range(3, 10):
            if 1 < self.arr_index_2_to_j[i - 3] < i - 1:
                self.arr_index_2_to_j[i - 3] += 1
                continue
            if self.arr_index_2_to_j[i - 3] >= i - 1:
                self.arr_index_2_to_j[i - 3] += 2

        self.Array_l_0 = np.zeros(7)
        self.Array_l_1 = np.zeros(7)
        self.Array_l_2 = np.zeros(7)
        for _ in range(7):
            self.Array_l_0[_] = self.array_l[0][_ + 2]
        for _ in range(7):
            self.Array_l_1[_] = self.array_l[1][_ + 2]
        for _ in range(7):
            self.Array_l_2[_] = self.array_l[2][_ + 2]

        # action space
        self.action_space = spaces.MultiDiscrete([8]*3)

        # state space
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(47,), dtype=np.float32)

        self.step_count = 0
        self.max_steps = 5
        self.max_time = 0

    def reset(self):
        self.step_count = 0
        i = np.random.randint(0,np.array(self.data_lib).shape[0])
        self.data = np.array(self.data_lib[i])
        self.state = self.data
        obs = self.state
        return obs

    def step(self, action):
        self.step_count += 1
        P_1, P_2, P_3 = action
        cost, delay = self._compute_cost(P_1, P_2, P_3)
        self.max_time = delay
        reward = -cost
        done = self.step_count == self.max_steps

        # update positions
        for i in range(17, len(self.state), 3):
            x, y, z = self.state[i], self.state[i + 1], self.state[i + 2]
            self.state[i] = x + self.max_time * z

        # update connectivity
        for j in range(17, 27):
            for i in range(j + 1, 27):
                if np.sqrt((self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 18] = 0
                    self.array_S[j - 17][i - 18] = 1000
                else:
                    self.array_S[j - 17][i - 18] = np.sqrt(
                        (self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                self.state[17 + 1 + 3 * (i - 17)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[17 + 2 + 3 * (i - 17)] / self.state[17 + 2 + 3 * (j - 17)] < 0:
                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) + self.state[17 + 3 * (j - 17)] - self.state[
                                                                17 + 3 * (i - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) -
                                                            self.state[17 + 3 * (i - 17)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[17 + 3 * (i - 17)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[17 + 2 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[
                                17 + 1 + 3 * (i - 17)]) ** 2) +
                                                            self.state[17 + 3 * (i - 17)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[17 + 2 + 3 * (i - 17)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[17 + 2 + 3 * (i - 17)] < 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[17 + 3 * (i - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (i - 17)] > self.state[17 + 2 + 3 * (j - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range - (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[17 + 2 + 3 * (i - 17)] > 0 and self.state[
                            17 + 3 * (i - 17)] > self.state[17 + 3 * (j - 17)]:
                            self.array_l[j - 17][i - 18] = (self.trans_range + (
                                    self.state[17 + 3 * (i - 17)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[17 + 2 + 3 * (i - 17)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

            for i in range(17, j):
                if np.sqrt((self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                        self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2) > self.trans_range:
                    self.array_l[j - 17][i - 17] = 0
                    self.array_S[j - 17][i - 17] = 1000
                else:
                    self.array_S[j - 17][i - 17] = np.sqrt(
                        (self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)]) ** 2 + (
                                self.state[47 + 1 + 3 * (i - 27)] - self.state[17 + 1 + 3 * (j - 17)]) ** 2)
                    if self.state[47 + 2 + 3 * (i - 27)] / self.state[17 + 2 + 3 * (j - 17)] < 0:
                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] < 0 < self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            - self.state[47 + 3 * (i - 27)] + self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)] and self.state[
                            17 + 2 + 3 * (j - 17)] > 0 > self.state[47 + 2 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (np.sqrt(self.trans_range ** 2 - (
                                    self.state[17 + 1 + 3 * (j - 17)] - self.state[47 + 1 + 3 * (i - 27)]) ** 2)
                                                            + self.state[47 + 3 * (i - 27)] - self.state[
                                                                17 + 3 * (j - 17)]) / (abs(
                                self.state[47 + 2 + 3 * (i - 27)]) + abs(self.state[17 + 2 + 3 * (j - 17)]))
                    else:
                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[47 + 2 + 3 * (i - 27)] < self.state[17 + 2 + 3 * (j - 17)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] < 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] > self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if 0 < self.state[17 + 2 + 3 * (j - 17)] < self.state[47 + 2 + 3 * (i - 27)] and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range - (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

                        if self.state[17 + 2 + 3 * (j - 17)] > self.state[47 + 2 + 3 * (i - 27)] > 0 and self.state[
                            17 + 3 * (j - 17)] < self.state[47 + 3 * (i - 27)]:
                            self.array_l[j - 17][i - 17] = (self.trans_range + (
                                    self.state[47 + 3 * (i - 27)] - self.state[17 + 3 * (j - 17)])) / abs(abs(
                                self.state[47 + 2 + 3 * (i - 27)]) - abs(self.state[17 + 2 + 3 * (j - 17)]))

        self.array_l_2_0 = np.zeros(7)
        self.array_l_2_1 = np.zeros(7)
        self.array_l_2_2 = np.zeros(7)

        arr_temp_0 = np.zeros(8)
        self.arr_index_0_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            for k in range(1, j):
                arr_temp_0[k - 1] = min(self.array_l[0][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_0[k - 2] = min(self.array_l[0][k - 1], self.array_l[k][j])

            self.array_l_2_0[j - 3] = max(arr_temp_0)
            self.arr_index_0_to_j[j - 3] = arr_temp_0.tolist().index(self.array_l_2_0[j - 3])

        for i in range(3, 10):
            if self.arr_index_0_to_j[i - 3] < i - 1:
                self.arr_index_0_to_j[i - 3] += 1
                continue
            if self.arr_index_0_to_j[i - 3] >= i - 1:
                self.arr_index_0_to_j[i - 3] += 2

        arr_temp_1 = np.zeros(8)
        self.arr_index_1_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_1[0] = min(self.array_l[1][0], self.array_l[0][j - 1])
            for k in range(2, j):
                arr_temp_1[k - 1] = min(self.array_l[1][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_1[k - 2] = min(self.array_l[1][k - 1], self.array_l[k][j])

            self.array_l_2_1[j - 3] = max(arr_temp_1)
            self.arr_index_1_to_j[j - 3] = arr_temp_1.tolist().index(self.array_l_2_1[j - 3])

        for i in range(3, 10):
            if 0 < self.arr_index_1_to_j[i - 3] < i - 1:
                self.arr_index_1_to_j[i - 3] += 1
                continue
            if self.arr_index_1_to_j[i - 3] >= i - 1:
                self.arr_index_1_to_j[i - 3] += 2

        arr_temp_2 = np.zeros(8)
        self.arr_index_2_to_j = np.zeros(7, dtype=int)

        for j in range(3, 10):
            arr_temp_2[0] = min(self.array_l[2][0], self.array_l[0][j - 1])
            arr_temp_2[1] = min(self.array_l[2][1], self.array_l[1][j - 1])
            for k in range(3, j):
                arr_temp_2[k - 1] = min(self.array_l[2][k - 1], self.array_l[k][j - 1])
            for k in range(j + 1, 10):
                arr_temp_2[k - 2] = min(self.array_l[2][k - 1], self.array_l[k][j])

            self.array_l_2_2[j - 3] = max(arr_temp_2)
            self.arr_index_2_to_j[j - 3] = arr_temp_2.tolist().index(self.array_l_2_2[j - 3])

        for i in range(3, 10):
            if 1 < self.arr_index_2_to_j[i - 3] < i - 1:
                self.arr_index_2_to_j[i - 3] += 1
                continue
            if self.arr_index_2_to_j[i - 3] >= i - 1:
                self.arr_index_2_to_j[i - 3] += 2

        self.Array_l_0 = np.zeros(7)
        self.Array_l_1 = np.zeros(7)
        self.Array_l_2 = np.zeros(7)
        for _ in range(7):
            self.Array_l_0[_] = self.array_l[0][_ + 2]
        for _ in range(7):
            self.Array_l_1[_] = self.array_l[1][_ + 2]
        for _ in range(7):
            self.Array_l_2[_] = self.array_l[2][_ + 2]

        obs = self.state

        return obs, float(reward), done, {}

    def _compute_cost(self, P_1, P_2, P_3):

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, c3, c4, c5, c6, c7, c8, c9, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = self.state

        c0_task = 0
        c1_task = 0
        c2_task = 0

        c3_task_0 = 0
        c3_task_1 = 0
        c3_task_2 = 0

        c4_task_0 = 0
        c4_task_1 = 0
        c4_task_2 = 0

        c5_task_0 = 0
        c5_task_1 = 0
        c5_task_2 = 0

        c6_task_0 = 0
        c6_task_1 = 0
        c6_task_2 = 0

        c7_task_0 = 0
        c7_task_1 = 0
        c7_task_2 = 0

        c8_task_0 = 0
        c8_task_1 = 0
        c8_task_2 = 0

        c9_task_0 = 0
        c9_task_1 = 0
        c9_task_2 = 0


        c3_trans_0 = 0
        c3_trans_1 = 0
        c3_trans_2 = 0

        c4_trans_0 = 0
        c4_trans_1 = 0
        c4_trans_2 = 0

        c5_trans_0 = 0
        c5_trans_1 = 0
        c5_trans_2 = 0

        c6_trans_0 = 0
        c6_trans_1 = 0
        c6_trans_2 = 0

        c7_trans_0 = 0
        c7_trans_1 = 0
        c7_trans_2 = 0

        c8_trans_0 = 0
        c8_trans_1 = 0
        c8_trans_2 = 0

        c9_trans_0 = 0
        c9_trans_1 = 0
        c9_trans_2 = 0


        c3_exe_0 = 0
        c3_exe_1 = 0
        c3_exe_2 = 0

        c4_exe_0 = 0
        c4_exe_1 = 0
        c4_exe_2 = 0

        c5_exe_0 = 0
        c5_exe_1 = 0
        c5_exe_2 = 0

        c6_exe_0 = 0
        c6_exe_1 = 0
        c6_exe_2 = 0

        c7_exe_0 = 0
        c7_exe_1 = 0
        c7_exe_2 = 0

        c8_exe_0 = 0
        c8_exe_1 = 0
        c8_exe_2 = 0

        c9_exe_0 = 0
        c9_exe_1 = 0
        c9_exe_2 = 0


        c3_delay_0 = 0
        c3_delay_1 = 0
        c3_delay_2 = 0

        c4_delay_0 = 0
        c4_delay_1 = 0
        c4_delay_2 = 0

        c5_delay_0 = 0
        c5_delay_1 = 0
        c5_delay_2 = 0

        c6_delay_0 = 0
        c6_delay_1 = 0
        c6_delay_2 = 0

        c7_delay_0 = 0
        c7_delay_1 = 0
        c7_delay_2 = 0

        c8_delay_0 = 0
        c8_delay_1 = 0
        c8_delay_2 = 0

        c9_delay_0 = 0
        c9_delay_1 = 0
        c9_delay_2 = 0

        punish = 0

        c_map = {
            1: c3,
            2: c4,
            3: c5,
            4: c6,
            5: c7,
            6: c8,
            7: c9
        }
        for val in range(1, 8):
            count = [P_1, P_2, P_3].count(val)
            if count == 2:
                c_map[val] /= 2
            elif count == 3:
                c_map[val] /= 3
        c3, c4, c5, c6, c7, c8, c9 = c_map[1], c_map[2], c_map[3], c_map[4], c_map[5], c_map[6], c_map[7]

        if P_1 == 0:
            c0_task += t1
        if P_2 == 0:
            c1_task += t2
        if P_3 == 0:
            c2_task += t3

        # cost in SV3
        if P_1 == 1:
            if self.Array_l_0[0] == 0:
                punish += 100
            else:
                c3_task_0 += t1
                c3_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][2] ** self.trans_loss) * self.noise)))
                c3_exe_0 += c3_task_0 / c3
                c3_delay_0 += c3_trans_0 + c3_exe_0
                if c3_delay_0 > self.Array_l_0[0]:
                    punish += 100

        if P_2 == 1:
            if self.Array_l_1[0] == 0:
                punish += 100
            else:
                c3_task_1 += t2
                c3_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][2] ** self.trans_loss) * self.noise)))
                c3_exe_1 += c3_task_1 / c3
                c3_delay_1 += c3_trans_1 + c3_exe_1
                if c3_delay_1 > self.Array_l_1[0]:
                    punish += 100

        if P_3 == 1:
            if self.Array_l_2[0] == 0:
                punish += 100
            else:
                c3_task_2 += t3
                c3_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][2] ** self.trans_loss) * self.noise)))
                c3_exe_2 += c3_task_2 / c3
                c3_delay_2 += c3_trans_2 + c3_exe_2
                if c3_delay_2 > self.Array_l_2[0]:
                    punish += 100

        c3_delay = max(c3_delay_0, c3_delay_1, c3_delay_2)
        c3_task = c3_task_0 + c3_task_1 + c3_task_2
        c3_cost = c3_task / self.intensity * self.computation_cost

        # cost in SV4
        if P_1 == 2:
            if self.Array_l_0[1] == 0:
                punish += 100
            else:
                c4_task_0 += t1
                c4_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][3] ** self.trans_loss) * self.noise)))
                c4_exe_0 += c4_task_0 / c4
                c4_delay_0 += c4_trans_0 + c4_exe_0
                if c4_delay_0 > self.Array_l_0[1]:
                    punish += 100

        if P_2 == 2:
            if self.Array_l_1[1] == 0:
                punish += 100
            else:
                c4_task_1 += t2
                c4_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][3] ** self.trans_loss) * self.noise)))
                c4_exe_1 += c4_task_1 / c4
                c4_delay_1 += c4_trans_1 + c4_exe_1
                if c4_delay_1 > self.Array_l_1[1]:
                    punish += 100

        if P_3 == 2:
            if self.Array_l_2[1] == 0:
                punish += 100
            else:
                c4_task_2 += t3
                c4_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][3] ** self.trans_loss) * self.noise)))
                c4_exe_2 += c4_task_2 / c4
                c4_delay_2 += c4_trans_2 + c4_exe_2
                if c4_delay_2 > self.Array_l_2[1]:
                    punish += 100

        c4_delay = max(c4_delay_0, c4_delay_1, c4_delay_2)
        c4_task = c4_task_0 + c4_task_1 + c4_task_2
        c4_cost = c4_task / self.intensity * self.computation_cost

        # cost in SV5
        if P_1 == 3:
            if self.Array_l_0[2] == 0:
                punish += 100
            else:
                c5_task_0 += t1
                c5_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][4] ** self.trans_loss) * self.noise)))
                c5_exe_0 += c5_task_0 / c5
                c5_delay_0 += c5_trans_0 + c5_exe_0
                if c5_delay_0 > self.Array_l_0[2]:
                    punish += 100

        if P_2 == 3:
            if self.Array_l_1[2] == 0:
                punish += 100
            else:
                c5_task_1 += t2
                c5_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][4] ** self.trans_loss) * self.noise)))
                c5_exe_1 += c5_task_1 / c5
                c5_delay_1 += c5_trans_1 + c5_exe_1
                if c5_delay_1 > self.Array_l_1[2]:
                    punish += 100

        if P_3 == 3:
            if self.Array_l_2[2] == 0:
                punish += 100
            else:
                c5_task_2 += t3
                c5_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][4] ** self.trans_loss) * self.noise)))
                c5_exe_2 += c5_task_2 / c5
                c5_delay_2 += c5_trans_2 + c5_exe_2
                if c5_delay_2 > self.Array_l_2[2]:
                    punish += 100

        c5_delay = max(c5_delay_0, c5_delay_1, c5_delay_2)
        c5_task = c5_task_0 + c5_task_1 + c5_task_2
        c5_cost = c5_task / self.intensity * self.computation_cost

        # cost in SV6
        if P_1 == 4:
            if self.Array_l_0[3] == 0:
                punish += 100
            else:
                c6_task_0 += t1
                c6_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][5] ** self.trans_loss) * self.noise)))
                c6_exe_0 += c6_task_0 / c6
                c6_delay_0 += c6_trans_0 + c6_exe_0
                if c6_delay_0 > self.Array_l_0[3]:
                    punish += 100

        if P_2 == 4:
            if self.Array_l_1[3] == 0:
                punish += 100
            else:
                c6_task_1 += t2
                c6_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][5] ** self.trans_loss) * self.noise)))
                c6_exe_1 += c6_task_1 / c6
                c6_delay_1 += c6_trans_1 + c6_exe_1
                if c6_delay_1 > self.Array_l_1[3]:
                    punish += 100

        if P_3 == 4:
            if self.Array_l_2[3] == 0:
                punish += 100
            else:
                c6_task_2 += t3
                c6_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][5] ** self.trans_loss) * self.noise)))
                c6_exe_2 += c6_task_2 / c6
                c6_delay_2 += c6_trans_2 + c6_exe_2
                if c6_delay_2 > self.Array_l_2[3]:
                    punish += 100

        c6_delay = max(c6_delay_0, c6_delay_1, c6_delay_2)
        c6_task = c6_task_0 + c6_task_1 + c6_task_2
        c6_cost = c6_task / self.intensity * self.computation_cost

        # cost in SV7
        if P_1 == 5:
            if self.Array_l_0[4] == 0:
                punish += 100
            else:
                c7_task_0 += t1
                c7_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][6] ** self.trans_loss) * self.noise)))
                c7_exe_0 += c7_task_0 / c7
                c7_delay_0 += c7_trans_0 + c7_exe_0
                if c7_delay_0 > self.Array_l_0[4]:
                    punish += 100

        if P_2 == 5:
            if self.Array_l_1[4] == 0:
                punish += 100
            else:
                c7_task_1 += t2
                c7_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][6] ** self.trans_loss) * self.noise)))
                c7_exe_1 += c7_task_1 / c7
                c7_delay_1 += c7_trans_1 + c7_exe_1
                if c7_delay_1 > self.Array_l_1[4]:
                    punish += 100

        if P_3 == 5:
            if self.Array_l_2[4] == 0:
                punish += 100
            else:
                c7_task_2 += t3
                c7_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][6] ** self.trans_loss) * self.noise)))
                c7_exe_2 += c7_task_2 / c7
                c7_delay_2 += c7_trans_2 + c7_exe_2
                if c7_delay_2 > self.Array_l_2[4]:
                    punish += 100

        c7_delay = max(c7_delay_0, c7_delay_1, c7_delay_2)
        c7_task = c7_task_0 + c7_task_1 + c7_task_2
        c7_cost = c7_task / self.intensity * self.computation_cost

        # cost in SV8
        if P_1 == 6:
            if self.Array_l_0[5] == 0:
                punish += 100
            else:
                c8_task_0 += t1
                c8_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][7] ** self.trans_loss) * self.noise)))
                c8_exe_0 += c8_task_0 / c8
                c8_delay_0 += c8_trans_0 + c8_exe_0
                if c8_delay_0 > self.Array_l_0[5]:
                    punish += 100

        if P_2 == 6:
            if self.Array_l_1[5] == 0:
                punish += 100
            else:
                c8_task_1 += t2
                c8_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][7] ** self.trans_loss) * self.noise)))
                c8_exe_1 += c8_task_1 / c8
                c8_delay_1 += c8_trans_1 + c8_exe_1
                if c8_delay_1 > self.Array_l_1[5]:
                    punish += 100

        if P_3 == 6:
            if self.Array_l_2[5] == 0:
                punish += 100
            else:
                c8_task_2 += t3
                c8_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][7] ** self.trans_loss) * self.noise)))
                c8_exe_2 += c8_task_2 / c8
                c8_delay_2 += c8_trans_2 + c8_exe_2
                if c8_delay_2 > self.Array_l_2[5]:
                    punish += 100

        c8_delay = max(c8_delay_0, c8_delay_1, c8_delay_2)
        c8_task = c8_task_0 + c8_task_1 + c8_task_2
        c8_cost = c8_task / self.intensity * self.computation_cost

        # cost in SV9
        if P_1 == 7:
            if self.Array_l_0[6] == 0:
                punish += 100
            else:
                c9_task_0 += t1
                c9_trans_0 += (t1 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[0][8] ** self.trans_loss) * self.noise)))
                c9_exe_0 += c9_task_0 / c9
                c9_delay_0 += c9_trans_0 + c9_exe_0
                if c9_delay_0 > self.Array_l_0[6]:
                    punish += 100

        if P_2 == 7:
            if self.Array_l_1[6] == 0:
                punish += 100
            else:
                c9_task_1 += t2
                c9_trans_1 += (t2 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[1][8] ** self.trans_loss) * self.noise)))
                c9_exe_1 += c9_task_1 / c9
                c9_delay_1 += c9_trans_1 + c9_exe_1
                if c9_delay_1 > self.Array_l_1[6]:
                    punish += 100

        if P_3 == 7:
            if self.Array_l_2[6] == 0:
                punish += 100
            else:
                c9_task_2 += t3
                c9_trans_2 += (t3 / self.intensity) / (
                        self.bandWith * np.log2(
                    1 + self.trans_power / ((self.array_S[2][8] ** self.trans_loss) * self.noise)))
                c9_exe_2 += c9_task_2 / c9
                c9_delay_2 += c9_trans_2 + c9_exe_2
                if c9_delay_2 > self.Array_l_2[6]:
                    punish += 100

        c9_delay = max(c9_delay_0, c9_delay_1, c9_delay_2)
        c9_task = c9_task_0 + c9_task_1 + c9_task_2
        c9_cost = c9_task / self.intensity * self.computation_cost

        c0_delay = c0_task / self.c0
        c1_delay = c1_task / self.c1
        c2_delay = c2_task / self.c2

        delay = max(c0_delay, c1_delay, c2_delay, c3_delay, c4_delay, c5_delay, c6_delay, c7_delay, c8_delay, c9_delay)

        cost = delay + self.wc*(c9_cost + c8_cost + c7_cost + c6_cost + c5_cost + c4_cost + c3_cost) + punish

        return cost, delay
'''
