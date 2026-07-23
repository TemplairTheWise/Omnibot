from gpiozero import DistanceSensor
from time import sleep

# Define your GPIO pins here (use the BCM numbering, not the physical pin numbers)
# Example: Trigger on GPIO 23, Echo on GPIO 24
TRIGGER_PIN = 5
ECHO_PIN = 6

# Initialize the sensor
# We set max_distance to 4 (meters) since the HC-SR04 can read up to ~400cm
sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN, max_distance=4.0)

print("Starting distance measurement... Press Ctrl+C to stop.")

try:
    while True:
        # The sensor returns the distance in meters. Multiply by 100 for cm.
        distance_cm = sensor.distance * 100
        
        # Print the distance formatted to one decimal place
        print(f"Distance: {distance_cm:.1f} cm")
        
        # Wait 1 second before the next reading
        sleep(1)

except KeyboardInterrupt:
    print("\nMeasurement stopped by user.")
finally:
    sensor.close()
