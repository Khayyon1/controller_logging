import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image
import numpy as np
import glob
import pygame as p

# Define the neural network model
class SonicControllerModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SonicControllerModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load the saved frames from pc
def load_frames(file_paths):
    frames = []
    for path in file_paths:
        img = Image.open(path)
        # Preprocess the image (resize, normalize, convert to tensor, etc)
        img = img.resize((64, 64)) # resize to match model input size
        img = np.array(img) / 255.0 # normalize pixels values [0, 1)
        img = np.transpose(img, (2, 0, 1)) #Convert to channel-first format
        frames.append(img)
    return np.array(frames)

print("Loading and Preprocessing the Frames")
frame_paths = [file for file in glob.glob("..\\vid2imgs\\output_images\\*.jpg")]
frames = load_frames(frame_paths)

# Sample data (replace with your actual data)
print("Intializing the model")
input_channels = 3 # assuming RGB images
num_classes = 5 # assuming 5 controller commands
model = SonicControllerModel(input_channels, num_classes)

frames_tensor = torch.tensor(frames, dtype=torch.float32)

# # Create PyTorch datasets and data loaders
# dataset = TensorDataset(frames, controller_commands)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # init the model, loss fn, and optimizer
# model = SonicControllerModel(input_channels, num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(params=model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 5
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(data_loader):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 10 == 9: # Print every 10 mini-batches
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {running_loss / 10:.4f}')
#             running_loss = 0.0

# print('Finished Training')

# Button mapping for PS4
ps4_full_layout = {
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

model_to_ps4 = {
    0: 14,
    1: 0,
    2: 13,
    3: 0,
    4: 14,
}

def convert_to_button(command):
    return ps4_full_layout[model_to_ps4[command]]

predicted_button_presses = []
predicted = []

# Example of using the trained model to predict controller commands for new frames
print("Running the model")
with torch.no_grad():
    outputs = model(frames_tensor)
    predicted_commands = torch.argmax(outputs, dim=1)
    predicted = predicted_commands.tolist()
    print('Predicted Controller Commands:', predicted_commands.tolist())
    predicted_button_presses = [convert_to_button(cmd) for cmd in predicted_commands.tolist()]
    print('Predicted buttons ', predicted_button_presses)
    
print("Sending the inputs to the controller")
# Need to have pygame controller use these inputs
p.init()
p.joystick.init()
joystick_count = p.joystick.get_count()
print(joystick_count)
if joystick_count > 0:
    joystick = p.joystick.Joystick(0)
    joystick.init()

import random
import pyautogui
p_copy = []

running = True
while running:
    for event in p.event.get():
        if event.type == p.QUIT or joystick.get_button(15) == True:
            running = False
    
    # Get predicted command
    if len(p_copy) == 0:
        random.shuffle(predicted)
        p_copy = predicted.copy()
    
    
    # JoyStick
    if joystick_count > 0 and len(p_copy) > 0:
        print(f"Length of predicted commands left: {len(p_copy)}")
        predicted_command = p_copy.pop(0)  # Use the first predicted command
        print(convert_to_button(predicted_command))
        # Map predicted command to joystick input
        if predicted_command == 0:
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_SPACE}))
            pyautogui.press('space')
            with pyautogui.hold('space'):
                pyautogui.sleep('space')


        elif predicted_command == 1:
            # Example: Move joystick down
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_LEFT}))
            pyautogui.keyDown('left')
        elif predicted_command == 2:
            # Example: Move joystick left
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_RIGHT}))
            pyautogui.keyDown('right')

        elif predicted_command == 3:
            # Example: Move joystick right
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_RIGHT}))
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_SPACE}))
            with pyautogui.hold('right'):
                pyautogui.keyDown('space')

        elif predicted_command == 4:
            # # Example: Press joystick button
            # joystick.set_button(0, 1)
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_LEFT}))
            p.event.post(p.event.Event(p.KEYDOWN, {'key': p.K_SPACE}))
            with pyautogui.hold('left'):
                pyautogui.keyDown('space')

    # # JoyStick
    # if joystick_count > 0 and len(p_copy) > 0:
    #     print(f"Length of predicted commands left: {len(p_copy)}")
    #     predicted_command = p_copy.pop(0)  # Use the first predicted command
    #     print(convert_to_button(predicted_command))
    #     # Map predicted command to joystick input
    #     if predicted_command == 0:
    #         p.event.post(p.event.Event(p.JOYBUTTONDOWN, {'joy': 0, 'button': 0}))

    #     elif predicted_command == 1:
    #         # Example: Move joystick down
    #         p.event.post(p.event.Event(p.JOYAXISMOTION, {'joy': 0, 'axis': 0, 'value': 1}))

    #     elif predicted_command == 2:
    #         # Example: Move joystick left
    #         p.event.post(p.event.Event(p.JOYAXISMOTION, {'joy': 0, 'axis': 0, 'value': -1}))

    #     elif predicted_command == 3:
    #         # Example: Move joystick right
    #         p.event.post(p.event.Event(p.JOYAXISMOTION, {'joy': 0, 'axis': 0, 'value': 1}))
    #         p.event.post(p.event.Event(p.JOYBUTTONDOWN, {'joy': 0, 'button': 0}))

    #     elif predicted_command == 4:
    #         # # Example: Press joystick button
    #         # joystick.set_button(0, 1)
    #         p.event.post(p.event.Event(p.JOYBUTTONDOWN, {'joy': 0, 'button': 14}))

    

# # Quit Pygame
