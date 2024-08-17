def compute_rectangle_given_center(x_center, y_center, width, height):
    x1 = x_center - width / 2
    y1 = y_center + height / 2

    x2 = x_center + width / 2
    y2 = y_center + height / 2

    x3 = x_center - width / 2
    y3 = y_center - height / 2

    x4 = x_center + width / 2
    y4 = y_center - height / 2

    return (x1, y1), (x2, y2), (x3, y3), (x4, y4)
