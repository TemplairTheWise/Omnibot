import sys
import tty
import termios
import select
import time
from omnibot import OmniBot
from adafruit_motor import servo  # Added to control the gripper

def get_key_timeout(timeout_seconds):
    """Waits for a keypress, but returns None if the timeout is reached."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # Wait for input, but timeout if nothing is pressed
        rlist, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
        if rlist:
            ch = sys.stdin.read(1)
        else:
            ch = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    print("Initializing OmniBot...")
    bot = OmniBot()
    
    # Initialize the gripper on Channel 7
    # Note: bot.pca accesses the PCA9685 board initialized in your omnibot.py script
    gripper = servo.Servo(bot.pca.channels[7])
    
    # --- GRIPPER TUNING VARIABLES ---
    # Standard microservos usually accept values between 0 and 180.
    # Adjust these numbers until your gripper opens and closes perfectly.
    gripper_open_angle = 0   
    gripper_closed_angle = 150
    
    # Track the current state of the gripper (start with it open)
    gripper_is_closed = False
    gripper.angle = gripper_open_angle
    
    # Set the wheel movement speed (0 to 100)
    speed = 100

    print("\r\n--- OmniBot Control (With Gripper & Heartbeat) ---")
    print("\rW: Forward | S: Backward | A: Left | D: Right")
    print("\rQ: Rotate Left | E: Rotate Right")
    print("\rG: Toggle Gripper Open/Closed")
    print("\rSpacebar: Stop | X: Exit")
    print("\r--------------------------------------------------\n")

    while True:
        # Wait up to 12 seconds for a keypress
        key = get_key_timeout(12.0)

        # If 12 seconds pass with no key pressed, trigger the heartbeat
        if key is None:
            # Send a 10% speed command
            bot.startMove([0, 1], 5) 
            # Hold it for just 50 milliseconds
            time.sleep(0.05) 
            # Instantly cut the power
            bot.stop()
            continue

        key = key.lower()

        if key == 'w':
            print("\rAction: Moving Forward      ", end="")
            bot.startMove([0, 1], speed)
        elif key == 's':
            print("\rAction: Moving Backward     ", end="")
            bot.startMove([0, -1], speed)
        elif key == 'a':
            print("\rAction: Moving Left         ", end="")
            bot.startMove([-1, 0], speed)
        elif key == 'd':
            print("\rAction: Moving Right        ", end="")
            bot.startMove([1, 0], speed)
        elif key == 'q':
            print("\rAction: Rotating Left       ", end="")
            bot.rotate("left", speed)
        elif key == 'e':
            print("\rAction: Rotating Right      ", end="")
            bot.rotate("right", speed)
        elif key == 'g':
            # Toggle the gripper state
            gripper_is_closed = not gripper_is_closed
            
            if gripper_is_closed:
                print("\rAction: Closing Gripper     ", end="")
                gripper.angle = gripper_closed_angle
            else:
                print("\rAction: Opening Gripper     ", end="")
                gripper.angle = gripper_open_angle
        elif key == ' ':
            print("\rAction: STOP                ", end="")
            bot.stop()
        elif key == 'x':
            print("\rExiting...                  \n")
            bot.stop()
            break

if __name__ == "__main__":
    main()