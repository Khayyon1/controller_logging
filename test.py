import pygame as p

p.joystick.init()
stick = p.joystick.Joystick(0)
print("initialized: ", bool(stick.get_init()))
print(stick.get_name())
print(stick.get_button(1))


