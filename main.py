from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mathMethods.mathFunctions import MathFunctions


class StateValues:
    def __init__(self):
        self.classes_learn_dataset: dict[str, dict[int, list[int]]] = {
            "A": {
                1: [21, 23, 2, 52, 11, 26, 22, 27, 29, 10],
                2: [29, 24, 3, 44, 13, 22, 26, 25, 22, 7],
                3: [22, 22, 4, 30, 19, 15, 38, 28, 27, 9],
                4: [27, 28, 5, 34, 16, 17, 32, 26, 25, 8],
                5: [25, 26, 6, 54, 12, 27, 24, 23, 28, 9],
                6: [21, 23, 8, 46, 11, 23, 22, 21, 26, 9],
                7: [28, 21, 9, 26, 17, 13, 34, 29, 23, 8],
                8: [26, 29, 10, 36, 18, 18, 36, 25, 21, 7],
                9: [23, 22, 10, 58, 13, 29, 26, 23, 29, 10],
                10: [21, 27, 11, 50, 11, 25, 22, 24, 23, 8],
                11: [29, 25, 13, 28, 15, 14, 30, 22, 24, 8],
                12: [28, 28, 14, 42, 16, 21, 32, 28, 22, 7],
                13: [23, 26, 15, 56, 19, 28, 38, 26, 28, 9],
                14: [24, 23, 16, 34, 17, 17, 34, 23, 26, 9],
                15: [22, 21, 17, 48, 12, 24, 24, 21, 23, 8],
            },
            "B": {
                1: [21, 29, 24, 24, 13, 12, 26, 29, 21, 7],
                2: [28, 23, 25, 12, 11, 6, 22, 22, 29, 10],
                3: [26, 24, 26, 18, 15, 9, 30, 27, 22, 7],
                4: [23, 22, 27, 8, 16, 4, 32, 25, 27, 9],
                5: [21, 28, 27, 28, 19, 14, 38, 28, 25, 8],
                6: [29, 26, 29, 24, 17, 12, 34, 27, 28, 9],
                7: [28, 23, 30, 16, 13, 8, 26, 25, 26, 9],
                8: [23, 21, 30, 30, 19, 15, 38, 28, 23, 8],
                9: [24, 29, 32, 4, 16, 2, 32, 26, 21, 7],
                10: [22, 22, 32, 22, 12, 11, 24, 23, 29, 10],
                11: [28, 27, 33, 12, 11, 6, 22, 27, 25, 8],
                12: [26, 25, 34, 26, 17, 13, 34, 25, 23, 8],
                13: [23, 28, 35, 6, 18, 3, 36, 28, 24, 8],
                14: [21, 26, 36, 20, 11, 10, 22, 26, 22, 7],
                15: [29, 23, 37, 12, 14, 6, 28, 23, 28, 9],
            },
            "C": {
                1: [22, 21, 20, 88, 17, 44, 34, 21, 26, 9],
                2: [27, 29, 21, 96, 13, 48, 26, 29, 23, 8],
                3: [25, 25, 22, 80, 19, 40, 38, 25, 21, 7],
                4: [28, 23, 23, 90, 16, 45, 32, 23, 29, 10],
                5: [26, 24, 23, 102, 12, 51, 24, 24, 22, 7],
                6: [23, 22, 25, 106, 11, 53, 22, 22, 27, 9],
                7: [21, 28, 26, 78, 17, 39, 34, 28, 25, 8],
                8: [29, 26, 26, 88, 18, 44, 36, 26, 28, 9],
                9: [25, 23, 26, 100, 11, 50, 22, 23, 29, 10],
                10: [27, 21, 27, 72, 14, 36, 28, 21, 22, 7],
                11: [28, 29, 29, 92, 15, 46, 30, 29, 27, 9],
                12: [26, 22, 30, 82, 13, 41, 26, 22, 25, 8],
                13: [23, 27, 30, 104, 19, 52, 38, 27, 28, 9],
                14: [21, 25, 32, 86, 17, 43, 34, 25, 26, 9],
                15: [29, 28, 32, 96, 12, 48, 24, 27, 23, 8],
            },
            "D": {
                1: [24, 29, 7, 70, 11, 35, 22, 25, 29, 10],
                2: [22, 22, 4, 72, 19, 36, 38, 28, 22, 7],
                3: [28, 27, 7, 74, 13, 37, 26, 26, 29, 10],
                4: [26, 25, 10, 74, 10, 37, 20, 23, 22, 7],
                5: [23, 28, 6, 78, 11, 39, 22, 21, 27, 9],
                6: [21, 26, 9, 78, 18, 39, 36, 29, 25, 8],
                7: [29, 23, 11, 80, 15, 40, 30, 25, 28, 9],
                8: [22, 21, 5, 82, 11, 41, 22, 23, 26, 9],
                9: [27, 29, 7, 84, 13, 42, 26, 24, 23, 8],
                10: [25, 25, 13, 86, 10, 43, 20, 22, 21, 7],
                11: [28, 23, 10, 88, 17, 44, 34, 28, 29, 10],
                12: [26, 24, 8, 90, 11, 45, 22, 26, 23, 8],
                13: [23, 22, 12, 92, 18, 46, 36, 23, 24, 8],
                14: [21, 28, 9, 96, 15, 48, 30, 21, 22, 7],
                15: [29, 26, 11, 100, 11, 50, 22, 29, 28, 7],
            }
        }
        self.Mathematics = MathFunctions(self.classes_learn_dataset)

    def get_corr_matrix(self):
        matrix = self.Mathematics.correlation_matrix()

        for i in matrix.values():
            print(i)  # Вывод матрицы корреляционных взаимосвязей

    def get_informative_signs(self):
        b_acd = self.Mathematics.informative_signs(
            [*self.classes_learn_dataset["B"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values(),
             *self.classes_learn_dataset["D"].values()]
        )
        # print(F"{b_acd}")

        d_ac = self.Mathematics.informative_signs(
            [*self.classes_learn_dataset["D"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values()]
        )
        # print(F"{d_ac}")
        a_c = self.Mathematics.informative_signs(
            [*self.classes_learn_dataset["A"].values()],
            [*self.classes_learn_dataset["C"].values()]
        )
        # print(F"{a_c}")

    def plots(self):
        # B - ACD

        # self.create_informative_plot(2, 5,
        #                                       [*self.classes_learn_dataset["B"].values()],
        #                                       [*self.classes_learn_dataset["A"].values(),
        #                                        *self.classes_learn_dataset["C"].values(),
        #                                        *self.classes_learn_dataset["D"].values()])

        self.create_informative_plot_with_func(
            2, 5,
            [*self.classes_learn_dataset["B"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values(),
             *self.classes_learn_dataset["D"].values()],
            "График отображения объектов классов B и не B(ACD) с ЛДФ",
            "Признак 2", "Признак 5",
            "Класс B", "Класс не B"
        )

        # self.create_informative_plot_with_func_jackknife(2, 5,
        #                                                  [*self.classes_learn_dataset["B"].values()],
        #                                                  [*self.classes_learn_dataset["A"].values(),
        #                                                   *self.classes_learn_dataset["C"].values(),
        #                                                   *self.classes_learn_dataset["D"].values()],
        #                                                  "График отображения объектов классов B и не B(ACD) с ЛДФ",
        #                                                  "Признак 2", "Признак 5",
        #                                                  "Класс B", "Класс не B")

        # D - AC
        #
        # self.create_informative_plot(2, 5,
        #                              [*self.classes_learn_dataset["D"].values()],
        #                              [*self.classes_learn_dataset["A"].values(),
        #                               *self.classes_learn_dataset["C"].values()])
        #
        # self.create_informative_plot_with_func(
        #     5, 2,
        #     [*self.classes_learn_dataset["D"].values()],
        #     [*self.classes_learn_dataset["A"].values(),
        #      *self.classes_learn_dataset["C"].values()],
        #     "График отображения объектов классов D и не D(AC) с ЛДФ",
        #     "Признак 2", "Признак 5",
        #     "Класс D", "Класс не D"
        # )
        # self.create_informative_plot_with_func_jackknife(2, 5,
        #                                                  [*self.classes_learn_dataset["D"].values()],
        #                                                  [*self.classes_learn_dataset["A"].values(),
        #                                                  *self.classes_learn_dataset["C"].values()],
        #                                                  "График отображения объектов классов D и не D(AC) с ЛДФ",
        #                                                  "Признак 2", "Признак 5",
        #                                                  "Класс D", "Класс не D")

        # A - C

        # self.create_informative_plot(3, 2,
        #                              [*self.classes_learn_dataset["A"].values()],
        #                              [*self.classes_learn_dataset["C"].values()])
        #
        # self.create_informative_plot_with_func(
        #     2, 3,
        #     [*self.classes_learn_dataset["A"].values()],
        #     [*self.classes_learn_dataset["C"].values()],
        #     "График отображения объектов классов A и не A(C) с ЛДФ",
        #     "Признак 2", "Признак 3",
        #     "Класс A", "Класс C"
        # )
        #
        # self.create_informative_plot_with_func_jackknife(2, 3,
        #                                                  [*self.classes_learn_dataset["A"].values()],
        #                                                      [*self.classes_learn_dataset["C"].values()],
        #                                                  "График отображения объектов классов A и не A(C) с ЛДФ",
        #                                                  "Признак 2", "Признак 3",
        #                                                  "Класс A", "Класс C")

    @staticmethod
    def solve_function_coefficients(vector_a: tuple,
                                    vector_b: tuple,
                                    ) -> tuple:
        vector_C_y = round(vector_a[0] - vector_b[0], 4)
        vector_C_x = round(vector_a[1] - vector_b[1], 4)
        a = round((vector_a[0] + vector_b[0]), 4)
        b = round((vector_a[1] + vector_b[1]), 4)
        c2 = (vector_C_y * a + vector_C_x * b) * (-0.5)
        coeff_2 = c2 / (vector_C_y * -1)
        coeff_1 = vector_C_x / (vector_C_y * -1)
        return coeff_1, coeff_2

    def devide_classes(self,
                       x: int, y: int,
                       y_class: list, not_class: list, ) -> tuple[list, list, list, list]:
        class_A_x = []
        class_A_y = []

        class_B_x = []
        class_B_y = []

        for elems in range(len(y_class)):
            class_A_x.append(y_class[elems][x])
            class_A_y.append(y_class[elems][y])

        for elems in range(len(not_class)):
            class_B_x.append(not_class[elems][x])
            class_B_y.append(not_class[elems][y])

        return class_A_x, class_A_y, class_B_x, class_B_y

    @staticmethod
    def average_values(class_A_x, class_A_y,  class_B_x, class_B_y):
        Xa = (round(sum(class_A_x) / len(class_A_x), 4), round(sum(class_A_y) / len(class_A_y), 4))
        Xb = (round(sum(class_B_x) / len(class_B_x), 4), round(sum(class_B_y) / len(class_B_y), 4))

        return Xa, Xb

    def create_informative_plot(self,
                                x: int,
                                y: int,
                                y_class: list,
                                not_class: list):
        """????"""
        class_A_x, class_A_y, class_B_x, class_B_y = self.devide_classes(x, y, y_class, not_class)

        plt.scatter(class_A_x, class_A_y, color='blue', label='Класс A')
        plt.scatter(class_B_x, class_B_y, color='red', label='Класс C')

        plt.title("График точек классов A и не A(C) по признакам")
        plt.ylabel("Признак 3")
        plt.xlabel("Признак 2")
        plt.legend()
        plt.show()

    def create_informative_plot_with_func(self,
                                          x: int, y: int,
                                          y_class: list, not_class: list,
                                          title: str,
                                          xlabel: str, ylabel: str,
                                          alabel: str, blabel: str,
                                          ):
        """????"""
        class_A_x, class_A_y, class_B_x, class_B_y = self.devide_classes(x, y, y_class, not_class)
        # if len(not_class) == 45:
        #     vector_a, vector_b = self.average_values(class_A_x, class_A_y, class_B_x, class_B_y)
        # else:
        vector_a, vector_b = self.average_values(class_A_y, class_A_x, class_B_y, class_B_x)
        print(vector_a, vector_b)

        plt.figure(figsize=(8, 6))

        plt.scatter(class_A_x, class_A_y, color='blue', label=alabel)
        plt.scatter(class_B_x, class_B_y, color='red', label=blabel)
        plt.scatter(vector_a[1], vector_a[0], color='orange', marker="D")
        plt.scatter(vector_b[1], vector_b[0], color='orange', marker="D")

        # (30.4667, 8.7333)(14.7556, 36.0222)
        # 1.7369184843836525 - 16.257177613916276

        plt.xlim(min(class_A_x + class_B_x) - 5, max(class_A_x + class_B_x) + 5)
        plt.ylim(min(class_A_y + class_B_y) - 5, max(class_A_y + class_B_y) + 5)

        x_values = np.linspace(min(class_A_x + class_B_x) - 10, max(class_A_x + class_B_x) + 10, 100)
        coef1, coef2 = self.solve_function_coefficients(vector_a, vector_b)
        y_values = coef1 * x_values + coef2
        print(coef1, coef2)

        plt.plot(x_values, y_values, color='black', linestyle='--', label='ЛДФ')

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid()
        plt.show()

    def create_informative_plot_with_func_jackknife(self,
                                                    x: int, y: int,
                                                    y_class: list, not_class: list,
                                                    title: str,
                                                    xlabel: str, ylabel: str,
                                                    alabel: str, blabel: str,
                                                    ):
        """????"""
        # Построение графика с ЛДФ
        class_A_x, class_A_y, class_B_x, class_B_y = self.devide_classes(x, y, y_class, not_class)
        vector_a, vector_b = self.average_values(class_A_y, class_A_x, class_B_y, class_B_x)

        plt.figure(figsize=(8, 6))

        plt.scatter(class_A_x, class_A_y, color='blue', label=alabel)
        plt.scatter(class_B_x, class_B_y, color='red', label=blabel)

        plt.xlim(min(class_A_x + class_B_x) - 5, max(class_A_x + class_B_x) + 5)
        plt.ylim(min(class_A_y + class_B_y) - 5, max(class_A_y + class_B_y) + 5)

        x_values = np.linspace(min(class_A_x + class_B_x) - 10, max(class_A_x + class_B_x) + 10, 100)
        coef1, coef2 = self.solve_function_coefficients(vector_a, vector_b)
        y_values = coef1 * x_values + coef2
        plt.plot(x_values, y_values, color='black', linestyle='--', label='ЛДФ', zorder=5)

        label_added = False

        # Добавление области неопределенности
        for a_elem in range(len(y_class)):
            temp_class_A_x = class_A_x.copy()
            temp_class_A_y = class_A_y.copy()
            temp_class_A_x.pop(a_elem)
            temp_class_A_y.pop(a_elem)
            for b_elem in range(len(not_class)):
                temp_class_B_x = class_B_x.copy()
                temp_class_B_y = class_B_y.copy()
                temp_class_B_x.pop(b_elem)
                temp_class_B_y.pop(b_elem)

                vector_jack_a, vector_jack_b = self.average_values(temp_class_A_y, temp_class_A_x, temp_class_B_y,
                                                                   temp_class_B_x)

                plt.xlim(min(temp_class_A_x + temp_class_B_x) - 5, max(temp_class_A_x + temp_class_B_x) + 5)
                plt.ylim(min(temp_class_A_y + temp_class_B_y) - 5, max(temp_class_A_y + temp_class_B_y) + 5)

                x_values = np.linspace(min(temp_class_A_x + temp_class_B_x) - 10,
                                       max(temp_class_A_x + temp_class_B_x) + 10, 100)
                coef1, coef2 = self.solve_function_coefficients(vector_jack_a, vector_jack_b)
                y_values = coef1 * x_values + coef2

                if label_added:
                    plt.plot(x_values, y_values, color='grey', linestyle='solid', zorder=1)
                else:
                    plt.plot(x_values, y_values, color='grey', linestyle='solid', zorder=1, label="О.Н.")
                    label_added = True

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    StateValues().plots()
    # StateValues().get_informative_signs()
    # print(StateValues().classes_learn_dataset.values())
