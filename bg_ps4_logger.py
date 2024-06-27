import pygame
import sys
import pandas as pd

ps4_mappings = {
    0: 'CROSS',
    1: 'CIRCLE',
    2: 'SQUARE',
    3: 'TRIANGLE',
    4: 'SHARE',
    5: 'PS BUTTON',
    6: 'OPTIONS',
    7: 'L3',
    8: 'R3',
    9: 'L1',
    10: 'R1',
    11: 'D-PAD UP',
    12: 'D-PAD DOWN',
    13: 'D-PAD LEFT',
    14: 'D-PAD RIGHT',
    # 15: 'TOUCHPAD'
}

LOG = []

def initialize():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick connected.")
        pygame.quit()
        sys.exit()

def log_controller_input():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        # Iterate over each connected joystick
        for i in range(pygame.joystick.get_count()):
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

            # # Log input for each axis
            # for axis in range(joystick.get_numaxes()):
            #     axis_value = joystick.get_axis(axis)
            #     print(f"Joystick {i}, Axis {axis}: {axis_value}")

            # Log input for each button
            for button in range(joystick.get_numbuttons()):
                button_value = joystick.get_button(button)
                
                if button in ps4_mappings.keys():
                    secondary = ps4_mappings[button]
                    if button_value != 0:
                        LOG.append(secondary)
                        print(f"Joystick {i}, {secondary} | Button {button}: {button_value}")
                else:
                    if button_value != 0:
                        print(f"Joystick {i}, Button {button}: {button_value}")
                        df = pd.DataFrame(LOG)
                        df.to_csv('ps4_log.csv')
                        pygame.quit()
                        sys.exit()

            # Log input for each hat
            # for hat in range(joystick.get_numhats()):
            #     hat_value = joystick.get_hat(hat)
            #     print(f"Joystick {i}, Hat {hat}: {hat_value}")

def main():
    initialize()
    log_controller_input()

if __name__ == "__main__":
    main()
