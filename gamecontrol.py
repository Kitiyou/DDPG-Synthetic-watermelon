"""与游戏进行简单交互"""

from selenium import webdriver
import time
import random

JS_FILE = 'gamecontrol.js'

# 模拟手机屏幕大小
VIEW_WIDTH = 375
VIEW_HEIGHT = 635
PIXEL_RATIO = 3.0

# 各种水果的半径
# TODO: 改为从网页中获取，因为某些改版游戏似乎有区别
FRUIT_RADIUS = [26, 39, 54, 59.5, 76, 91.5, 93, 129, 154, 214, 202]
NEXT_FRUIT_KIND_NUM = 5


class GameBox:

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class Fruit:

    def __init__(self, kind, x=None, y=None):
        self.kind = kind
        self.x = x
        self.y = y

    @property
    def radius(self):
        return FRUIT_RADIUS[self.kind]

    def __repr__(self):
        return '<Fruit %d (%.2f, %.2f)>' % (self.kind, self.x, self.y)


class Game:

    def __init__(self, url, browser):
        if browser.lower() != 'chrome':
            raise ValueError('暂不支持Chrome以外的浏览器：' % browser)
        self.driver = webdriver.Chrome(options=self._mobile_options())

        driver = self.driver
        driver.set_window_size(VIEW_WIDTH, VIEW_HEIGHT + 150)
        driver.get(url)

        # 等待加载完成
        while True:
            time.sleep(0.5)
            total_frames = driver.execute_script(
                'gf = __require("GameFunction"); '
                'return Boolean(gf.default.Instance && gf.default.Instance.targetFruit);'
            )
            if total_frames:
                break

        with open(JS_FILE, mode='r', encoding='utf-8') as f:
            driver.execute_script(f.read())

    def close(self):
        driver = self.driver
        driver.close()

    def skip_steps(self, min_steps_num):
        """使游戏时间快进多步，游戏默认1步为1/60秒"""
        self.driver.execute_script('window.watermelonAI.skipSteps({})'.format(min_steps_num))

    def restart(self):
        self.driver.execute_script('window.watermelonAI.restart();')

    def is_over(self):
        return self.driver.execute_script('return window.watermelonAI.isOver();')

    def get_score(self):
        # TODO: 死亡后剩余水果加的分数会被延迟加到总分，可以不等延迟自己给这部分分数加入进去
        return self.driver.execute_script('return window.watermelonAI.getScore();')

    def get_box_fruits(self):
        """获取所有已放置水果信息"""
        fruit_lists = self.driver.execute_script('return window.watermelonAI.getBoxFruits();')
        return [Fruit(*fruit_list) for fruit_list in fruit_lists]

    def get_next_fruit(self):
        """获取即将放置的水果，等待中则返回None"""
        kind = self.driver.execute_script('return window.watermelonAI.getNextFruit();')
        return Fruit(kind)

    def drop_fruit(self, x):
        """放置水果"""
        # TODO: 返回是否放置成功
        self.driver.execute_script('return window.watermelonAI.dropFruit({});'.format(x))

    def get_game_box(self):
        """获取场景宽高坐标信息，最上方为游戏结束判定虚线高度"""
        x_min = -360
        x_max = 360
        y_min = -1019.689
        y_max = self.driver.execute_script('return window.watermelonAI.getLineY();')
        return GameBox(x_min, x_max, y_min, y_max)

    @staticmethod
    def _mobile_options():
        mobile_emulation = {"deviceMetrics": {"width": VIEW_WIDTH, "height": VIEW_HEIGHT, "pixelRatio": PIXEL_RATIO}}
        options = webdriver.ChromeOptions()
        options.add_experimental_option("mobileEmulation", mobile_emulation)
        return options


def test():
    """测试程序
    Usage: gamecontrol url
    """
    import sys
    if len(sys.argv) < 2:
        print(test.__doc__)
        sys.exit(0)

    g = Game(url=sys.argv[1], browser='Chrome')
    box = g.get_game_box()
    while True:
        g.skip_steps(120)  # 开始游戏后生成新水果需要等待一段时间
        while not g.is_over():
            next_fruit = g.get_next_fruit()
            fruits = g.get_box_fruits()
            same_fruits = list(filter(lambda fruit: fruit.kind == next_fruit.kind, fruits))
            if same_fruits:
                x = max(same_fruits, key=lambda fruit: fruit.y).x
            else:
                x = random.uniform(
                    box.x_min + next_fruit.radius,
                    box.x_max - next_fruit.radius
                )
            g.drop_fruit(x)
            # time.sleep(2)
            g.skip_steps(120)  # 放置水果后生成新水果需要等待一段时间
        print('score1:', g.get_score())
        g.skip_steps(120)  # 判定游戏结束后，超过界限的水果闪烁需要一段时间
        time.sleep(4)  # 判定游戏结束后，剩余水果加分需要一段时间，因为用的是setTimeout所以目前没法用skip_steps加速
        print('score2:', g.get_score())

        g.restart()


if __name__ == '__main__':
    test()
