import pygame
import pandas as pd

pygame.init()


# This is a simple class that will help us print to the screen.
# It has nothing to do with the joysticks, just outputting the
# information.
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 25)

    def tprint(self, screen, text):
        text_bitmap = self.font.render(text, True, (0, 0, 0))
        screen.blit(text_bitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

pg2_ps4_controller_layout = {
    0: 'X Button',
    1: 'O Button',
    2: 'Square Button',
    3: 'Triangle Button',
    4: 'Select',
    5: 'PS Button',
    6: 'Start',
    7: 'L3',
    8: 'R3',
    9: 'L1',
    10: 'R1',
    11: 'D-Pad Up',
    12: 'D-Pad Down',
    13: 'D-Pad Left',
    14: 'D-Pad Right',
    15: 'Touch Pad'
}

# For L2, R2, JoySticks
pg2_ps4_controller_axis_layout = {
    0: 'L-Stick (x-axis)',
    1: 'L-Stick (y-axis)',
    2: 'R-Stick (x-axis)',
    3: 'R-Stick (y-axis)',
    4: 'L2',
    5: 'R2'
}

button_log = []

def main():
    # Set the width and height of the screen (width, height), and name the window.
    screen = pygame.display.set_mode((500, 700))
    pygame.display.set_caption("Joystick example")

    # Used to manage how fast the screen updates.
    clock = pygame.time.Clock()

    # Get ready to print.
    text_print = TextPrint()

    # This dict can be left as-is, since pygame will generate a
    # pygame.JOYDEVICEADDED event for every joystick connected
    # at the start of the program.
    joysticks = {}
    button_pressed = False
    done = False
    while not done:
        # Event processing step.
        # Possible joystick events: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
        # JOYBUTTONUP, JOYHATMOTION, JOYDEVICEADDED, JOYDEVICEREMOVED
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                done = True  # Flag that we are done so we exit this loop.

            if event.type == pygame.JOYBUTTONDOWN:
                if joystick == joysticks[0]:
                    print("Joystick button pressed.")
                    button_pressed = True

                    if event.button == 0:
                        joystick = joysticks[event.instance_id]
                        if joystick.rumble(0, 0.7, 500) and event.instance_id == 0:
                            print(f"Rumble effect played on joystick {event.instance_id}")

            if event.type == pygame.JOYBUTTONUP and joystick == joysticks[0]:
                    print("Joystick button released.")
                    button_pressed = False


            # Handle hotplugging
            if event.type == pygame.JOYDEVICEADDED:
                # This event will be generated when the program starts for every
                # joystick, filling up the list without needing to create them manually.
                joy = pygame.joystick.Joystick(event.device_index)
                joysticks[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connencted")

            if event.type == pygame.JOYDEVICEREMOVED:
                del joysticks[event.instance_id]
                print(f"Joystick {event.instance_id} disconnected")

        # Drawing step
        # First, clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
        screen.fill((255, 255, 255))
        text_print.reset()

        # Get count of joysticks.
        joystick_count = pygame.joystick.get_count()

        text_print.tprint(screen, f"Number of joysticks: {joystick_count}")
        text_print.indent()

        # For each joystick:
        pygame.joystick.init()
        stick = pygame.joystick.Joystick(0)
        for joystick in [stick]:
            jid = joystick.get_instance_id()

            text_print.tprint(screen, f"Joystick {jid}")
            text_print.indent()

            # Get the name from the OS for the controller/joystick.
            name = joystick.get_name()
            text_print.tprint(screen, f"Joystick name: {name}")

            guid = joystick.get_guid()
            text_print.tprint(screen, f"GUID: {guid}")

            power_level = joystick.get_power_level()
            text_print.tprint(screen, f"Joystick's power level: {power_level}")

            # Usually axis run in pairs, up/down for one, and left/right for
            # the other. Triggers count as axes.
            axes = joystick.get_numaxes()
            text_print.tprint(screen, f"Number of axes: {axes}")
            text_print.indent()

            for i in range(axes):
                axis = joystick.get_axis(i)
                secondary_name = ''
                if i in pg2_ps4_controller_axis_layout.keys():
                    secondary_name = pg2_ps4_controller_axis_layout[i]
                text_print.tprint(screen, f"({secondary_name}) Axis {i} value: {axis:>6.3f}")
                if button_pressed == True and jid==0:
                    # add logic to determine what counts as a button press for L2 AND R2
                    # add logic for what counts as valid analog stick mvmt
                    if (('L2' in secondary_name and axis > -0.5)\
                    or ('R2' in secondary_name and axis > -0.5)) and joystick == joysticks[0]:
                        button_log.append([secondary_name, i, axis])
            text_print.unindent()

            buttons = joystick.get_numbuttons()
            text_print.tprint(screen, f"Number of buttons: {buttons}")
            text_print.indent()

            for i in range(buttons):
                button = joystick.get_button(i)
                secondary_name = ""
                if i in pg2_ps4_controller_layout.keys():
                    secondary_name = pg2_ps4_controller_layout[i]
                text_print.tprint(screen, f"Button {i:>2} ({secondary_name}) value: {button} ")
                
                if button_pressed == True and button != 0 and jid==0:
                    button_log.append([secondary_name, i, button])
            text_print.unindent()

            hats = joystick.get_numhats()
            text_print.tprint(screen, f"Number of hats: {hats}")
            text_print.indent()

            # Hat position. All or nothing for direction, not a float like
            # get_axis(). Position is a tuple of int values (x, y).
            for i in range(hats):
                hat = joystick.get_hat(i)
                text_print.tprint(screen, f"Hat {i} value: {str(hat)}")
            text_print.unindent()

            text_print.unindent()

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # Limit to 60 frames per second.
        clock.tick(2)


if __name__ == "__main__":
    main()
    # If you forget this line, the program will 'hang'
    # on exit if running from IDLE.
    pygame.quit()
    
    df = pd.DataFrame(button_log, columns=['button_name','axis_number','value'])
    df.to_csv('ps4_button_log.csv')
