def find_dividing_line(center_a, center_b):
    # Координаты центров классов
    x1, y1 = center_a
    x2, y2 = center_b

    # Середина между центрами классов
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2

    # Наклон линии, соединяющей центры классов
    if x2 != x1:  # Чтобы избежать деления на ноль
        k_ab = (y2 - y1) / (x2 - x1)
        # Наклон разделяющей линии, перпендикулярной k_ab
        k_dividing = -1 / k_ab
    else:
        # Если точки имеют одинаковую x-координату, то разделяющая прямая вертикальная
        return "x = " + str(x_mid)

    # Вычисление свободного члена b
    b = y_mid - k_dividing * x_mid

    return k_dividing, b

point1 = (1, 2)
point2 = (3, 4)

k, b = find_dividing_line(point1, point2)
print(f"Уравнение прямой: y = {k} * x + {b}")