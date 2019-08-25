import pygame
from pygame.sprite import Sprite


class Ship(Sprite):
    # def中传递的是对象，实例对象（面向对象编程，什么都是对象）
    def __init__(self, ai_settings, screen):
        '''初始化飞船并设置其初始位置'''
        super().__init__()
        self.screen = screen
        self.ai_settings = ai_settings

        # 加载飞船图像并获取其外接矩形self.image = pygame.image.load(r'D:\python3wp\alien_invasion\images\ship.bmp')
        self.image = pygame.image.load(r'F:\PYTHON\alien_invasion\images\ship.bmp')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()

        # 将每艘新飞船放在屏幕底部中央
        self.rect.centerx = self.screen_rect.centerx
        self.rect.bottom = self.screen_rect.bottom

        # 在飞船的属性centerx中存储小数值
        self.center = float(self.rect.centerx)

        # 移动标志
        self.moving_right = False
        self.moving_left = False

    def update(self):
        '''根据移动标志调整飞船位置'''
        # 更新飞船的center值，而不是rect
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.center += self.ai_settings.ship_speed_factor
        if self.moving_left and self.rect.left > 0:
            self.center -= self.ai_settings.ship_speed_factor

        # 根据self.center更新rect对象,
        # self.rect.centerx 将只存储self.center 的整数部分，但对显示飞船而言，这问题不大
        self.rect.centerx = self.center

    # 方法中使用自定义i、常量、或__init__中的对象属性.xxx
    def blitme(self):
        '''在指定位置绘制飞船'''
        self.screen.blit(self.image, self.rect)

    def center_ship(self):
        '''让飞船在屏幕上居中'''
        self.center = self.screen_rect.centerx