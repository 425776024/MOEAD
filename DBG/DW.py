import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Draw(object):
    bound_x = []
    bound_y = []

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.plt = plt
        self.set_font()

    def draw_line(self, p_from, p_to):
        line1 = [(p_from[0], p_from[1]), (p_to[0], p_to[1])]
        (line1_xs, line1_ys) = zip(*line1)
        self.ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))

    # def draw_arrow(self, p_from, p_to):
    #     if p_from.shape[0] != 2 and p_to.shape[0] != 2:
    #         print('error,', p_from, p_to)
    #         return
    #     p_from = list(p_from)
    #     p_to = list(p_to)
    #     self.ax.arrow(p_from[0], p_from[1], p_to[0] - p_from[0], p_to[1] - p_from[1],
    #                   length_includes_head=True,
    #                   head_width=(self.bound_x[1] - self.bound_x[0]) / 100,
    #                   head_length=(self.bound_x[1] - self.bound_x[0]) / 50,
    #                   fc='blue', ec='black')

    def draw_points(self, pointx, pointy):
        self.ax.plot(pointx, pointy, 'ro')

    def set_xybound(self, x_bd, y_bd):
        self.ax.axis([x_bd[0], x_bd[1], y_bd[0], y_bd[1]])

    def draw_text(self, x, y, text, size=8):
        self.ax.text(x, y, text, fontsize=size)

    def set_font(self, ft_style='SimHei'):
        plt.rcParams['font.sans-serif'] = [ft_style]  # 用来正常显示中文标签
