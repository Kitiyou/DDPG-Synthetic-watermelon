import numpy as np
import math

import gamecontrol


STEP_NUM = 180  # 游戏跳过步数
IMG_WIDTH = 64  # 状态图像宽度


class Environment:

    def __init__(self, url, browser):
        self.game = gamecontrol.Game(url=url, browser=browser)
        self._game_box = self.game.get_game_box()
        self._calc_img_size()
        self.score = 0
        self._fruits_kind_num = len(gamecontrol.FRUIT_RADIUS)

    def _calc_img_size(self):
        """计算图片大小，并计算坐标转换用参数"""
        game_box = self._game_box
        self._scale = (IMG_WIDTH - 1) / (game_box.x_max - game_box.x_min)
        self._y_offset = - game_box.y_max * -self._scale
        self._x_offset = - game_box.x_min * self._scale
        img_height = math.ceil(game_box.y_min * -self._scale + self._y_offset) + 1
        self._img_size = (img_height, IMG_WIDTH)

    def _to_img_space(self, x, y, r):
        """游戏内坐标转为图片中坐标"""
        return x * self._scale + self._x_offset, y * -self._scale + self._y_offset, r * self._scale

    def get_state_shape(self):
        return self._img_size + (self._fruits_kind_num,), (gamecontrol.NEXT_FRUIT_KIND_NUM,)

    @staticmethod
    def _draw_circle(img, x0, y0, r):
        # 确定圆圈所占方形范围
        top = max(0, math.floor(y0 - (r - 0.5)))
        bottom = min(img.shape[0] - 1, math.ceil(y0 + (r - 0.5)))
        left = max(0, math.floor(x0 - (r - 0.5)))
        right = min(img.shape[1] - 1, math.ceil(x0 + (r - 0.5)))

        if bottom <= top or right <= left:
            return

        x_linspace = np.linspace(left - x0, right - x0, right - left + 1)  # dx = (x - x0)[right:left+1]
        y_linspace = np.linspace(top - y0, bottom - y0, bottom - top + 1)  # dy = (y - y0)[top:bottom+1]
        dx, dy = np.meshgrid(x_linspace, y_linspace)
        d = np.sqrt(dx**2 + dy**2)  # d = sqrt((x - x0)^2 + (y - y0)^2)
        circle = np.clip(r + 0.5 - d, 0., 1.)  # d=r时像素值为0.5，d=r+0.5时像素值为0

        img[top:bottom+1, left:right+1] += circle

    def _game_state(self):
        fruits = self.game.get_box_fruits()
        next_fruit = self.game.get_next_fruit()

        # 已放置的水果
        height, width, ch = self.get_state_shape()[0]
        imgs = np.zeros((ch, height, width), dtype='float32')
        for fruit in fruits:
            x, y, r = self._to_img_space(fruit.x, fruit.y, fruit.radius)
            self._draw_circle(imgs[fruit.kind], x, y, r)
        img = imgs.transpose((1, 2, 0))  # 多张图片转为多通道图片

        # 即将放置的水果
        next_fruit_one_hot = np.zeros(self.get_state_shape()[1], dtype='float32')
        next_fruit_one_hot[next_fruit.kind] = 1.

        return img, next_fruit_one_hot

    def reset(self):
        """转为初始状态。返回状态"""
        self.game.restart()
        self.game.skip_steps(STEP_NUM)
        self.score = 0
        return self._game_state()

    def step(self, action):
        """放置水果，动作空间[-1, 1]。返回状态、奖励、是否结束"""
        game_box = self._game_box
        x = (game_box.x_max * (action + 1) - game_box.x_min * (action - 1)) / 2
        self.game.drop_fruit(x)
        self.game.skip_steps(STEP_NUM)

        new_score = self.game.get_score()
        delta_score = new_score - self.score
        self.score = new_score

        done = self.game.is_over()

        return self._game_state(), delta_score, done


def test():
    """测试程序
    Usage: environment url
    """
    import sys
    import matplotlib.pyplot as plt
    if len(sys.argv) < 2:
        print(test.__doc__)
        sys.exit(0)

    env = Environment(url=sys.argv[1], browser='Chrome')

    scores = []

    for episode in range(100):
        env.reset()

        while True:
            action = np.random.uniform(-1, 1)
            next_state, reward, done = env.step(action)

            if done:
                scores.append(env.game.get_score())
                print('Episode {}: {}'.format(episode, scores[-1]))
                break

    print(scores)


if __name__ == '__main__':
    test()
