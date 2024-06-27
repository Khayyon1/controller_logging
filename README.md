# Notice my Controller

I got curious about getting my python programs to notice and log input from my PS4 controller. This could be for generating data for other projects. Also besides making abstract key mappers in the past to practice OOP and Design patterns in college. I never tried to leverage existing programs like DS4Windows in any of my scripts. So in this project I am going to try to leverage PyGame to register the input from the controller and generate an output that corresponds to the button press.

# How did it start?

After looking at alternatives like inputs packages, I settled on PyGame since there were
resources available and confirmation that the DS4Windows work with it. The PyGame script
also had a sample script that creates a simple UI to map to the button presses.

# Issues that occured

I am working on labeling the buttons in more readable format vs (axis or button 1 - 9).
The laptop is picking up multiple controller inputs. so if I press all the buttons on a ps4 controller (i.e. square, triangle, circle, cross) and expect 4 inputs, I'll probably have 12 or more inputs.
