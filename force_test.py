import time
from omnibot import OmniBot

print("Waking up OmniBot...")
bot = OmniBot()

print("Sending power to microservo on Channel 0...")
# Command the servo to move to 120 degrees
bot.motors[0].angle = 120 

time.sleep(2)

print("Returning to center...")
# Command the servo back to the middle (90 degrees)
bot.motors[0].angle = 90

print("Test complete.")
