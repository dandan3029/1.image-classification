""" dynamically load settings

author baiyu
"""
import conf.global_settings as settings
# 导入global_settings.py文件

class Settings:
    # 定义Settings类
    def __init__(self, settings):
        # 定义Settings类的初始化函数
        for attr in dir(settings):
            # dir() 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；
            # 带参数时，返回参数的属性、方法列表。如果参数包含方法__dir__()，该方法将被调用。
            # 如果参数不包含__dir__()，该方法将最大限度地收集参数信息。
            if attr.isupper():
                # 检测字符串中所有的字母是否都为大写
                setattr(self, attr, getattr(settings, attr))
                # 设置属性

settings = Settings(settings)
# 定义settings对象