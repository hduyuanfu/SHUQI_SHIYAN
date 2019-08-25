import pygame
from settings import Settings
from ship import Ship
import game_functions as gf
from pygame.sprite import Group
from game_stats import GameStats
from scoreboard import Scoreboard
from button import Button


def run_game():
    # 初始化pygame、设置和屏幕对象
    pygame.init()
    # 设置类实例化
    ai_settings = Settings()
    screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption('Alien Invasion')

    # 创建play按钮
    play_button = Button(ai_settings, screen, 'Play')

    # 创建一个用于存储游戏统计信息的实例,并创建一个记分牌实例，并将
    # 该实例传递给update_screen()
    stats = GameStats(ai_settings)
    sb = Scoreboard(ai_settings, screen, stats)

    # 创建一艘飞船
    ship = Ship(ai_settings, screen)

    # 创建一个用于存储子弹和外星人的编组
    bullets = Group()
    aliens = Group()

    # 创建外星人群,由于我们不再在alien_invasion.py中直接创建外星人，因此无需在这个文件中导入Alien 类
    gf.create_fleet(ai_settings, screen, ship, aliens)

    # 开始游戏的主循环
    while True:
        # 监视键盘和鼠标事件
        gf.check_events(ai_settings, screen, stats, sb, play_button, ship, aliens, bullets)
        if stats.game_active:
            ship.update()
            gf.update_bullets(ai_settings, screen, stats, sb, ship, aliens, bullets)
            gf.update_aliens(ai_settings, screen, stats, sb, ship, aliens, bullets)
        # 每次循环时都重绘屏幕，用背景色填充屏幕
        gf.update_screen(ai_settings, screen, stats, sb, ship, aliens, bullets, play_button)

run_game()