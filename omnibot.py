import math as m
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from board import SCL, SDA

class OmniBot:
    """
    Class for interfacing with omni wheeled robot
    """

    def __init__(self):
        self.i2c = busio.I2C(SCL, SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50

        self.motors = [
            servo.Servo(self.pca.channels[0]), #front left
            servo.Servo(self.pca.channels[1]), #front right
            servo.Servo(self.pca.channels[2]), #rear left
            servo.Servo(self.pca.channels[3])  #rear right
        ]

    def __setMotor(self, index: int, speedPercent: float) -> None:
        """
        Sets speed <speedPercent> of motor from list self.motors at index <index>

        Arguments:
            index (int): The index of the motor in the motor array
            speedPercent (float): The target speed for the motor in percents (-100 <= speedPercent <= 100)
        Returns:
            None
        """
        speed = (speedPercent / 100) * 90

        # Flip speed for mirrored motors
        if index % 2 == 0:
            speed = speed * -1

        # print(f"Setting {index} to {angle} ({speedPercent}%)")
        self.motors[index].angle = 90 + speed

    def startMove(self, vect: list[int], speed: float) -> None:
        """
        Starts moving the robot in the direction of the vector <vect> at the speed of <speed>.

        Arguments:
            vect (list[int]): Vector in format [x, y] which determines the direction of movement
            speed (float): Determines the speed of movement. (-100 <= speed <= 100)
        Returns:
            None
        """
        if speed > 100 or speed < -100:
            return

        angle = m.atan2(abs(vect[0]), abs(vect[1]))

        magnitude = (1 / m.sin(m.radians(180 - 45 - m.degrees(angle)))) * m.sin(m.radians(45)) * speed

        velX = m.cos(angle) * magnitude
        velY = m.sin(angle) * magnitude

        if (vect[1] < 0): 
            velX *= -1
        if (vect[0] < 0): 
            velY *= -1

        # print(f"Angle: {m.degrees(angle)}deg, Mag: {magnitude} Speed: {speed}%, VelX: {velX}%, VelY: {velY}%")

        # Front wheels
        self.__setMotor(0, velX + velY)
        self.__setMotor(1, velX - velY)

        # Back wheels
        self.__setMotor(2, velX - velY)
        self.__setMotor(3, velX + velY)

    def rotate(self, dir: str, speed: float) -> None:
        """
        Rotates the robot in a direction of <dir> at speed <speed>.

        Arguments:
            dir (str): direction of rotation (dir = right or dir = left)
            speed (float): Speed of rotation (0 <= speed <= 100)
        Returns:
            None
        """
        x = 1
        if dir == "left":
            x = -1
        
        # Left wheels
        self.__setMotor(0, speed*x)
        self.__setMotor(2, speed*x)

        # Right wheels
        self.__setMotor(1, speed*x*-1)
        self.__setMotor(3, speed*x*-1)

    def stop(self) -> None:
        """
        Stops the robot's movement
        Returns:
            None
        """
        # Set speed of every motor to 0
        self.__setMotor(0, 8)
        self.__setMotor(2,12)
        self.__setMotor(1,-8)
        self.__setMotor(3,-8)
