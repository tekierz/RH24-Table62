import serial
import serial.tools.list_ports
import time


def find_esp32(baud_rate=115200, timeout=1):
    """
    Scan all serial ports and attempt to connect to an ESP32.
    Returns the serial object if successful, or None if not found.
    """
    print("Scanning available serial ports...")
    ports = serial.tools.list_ports.comports()

    for port in ports:
        print(f"Trying port: {port.device}")
        try:
            # Attempt to open the port
            ser = serial.Serial(port.device, baudrate=baud_rate, timeout=timeout)
            time.sleep(2)  # Give the ESP32 time to reset (if required)

            # Send a test command to check the device
            ser.write(b'ping\n')  # Adjust command based on ESP32 firmware
            time.sleep(0.1)  # Wait for a response
            response = ser.read_all().decode('utf-8').strip()

            if response:  # If we receive a response, it's likely the ESP32
                print(f"ESP32 found on {port.device}: {response}")
                return ser
            else:
                ser.close()

        except (serial.SerialException, UnicodeDecodeError) as e:
            print(f"Failed to connect on {port.device}: {e}")

    print("No ESP32 found.")
    return None

def main():
    esp = find_esp32()
    if esp:
        print("Connected to ESP32. Ready for commands.")
        while True:
            command = input("Enter command (or 'exit' to quit): ")
            if command.lower() == "exit":
                break
            esp.write(command.encode('utf-8') + b'\n')
            response = esp.read_all().decode('utf-8').strip()
            print(f"Response: {response}")
        esp.close()
    else:
        print("Could not find ESP32. Exiting.")


esp = find_esp32()
def send_serial(str):
    global esp

    if esp:
        esp.write(str.encode('utf-8') + b'\n')
        response = esp.read_all().decode('utf-8').strip()
    else:
        esp = find_esp32()
send_serial('black')

if __name__ == "__main__":
    print('MAINMNAIM MIAINMAINMNAIM MIAINMAINMNAIM MIAINMAINMNAIM MIAINMAINMNAIM MIAIN')
    main()








# import serial
# import time

# # Configuration for the serial connection
# PORT = "COM3"  # Replace with your ESP32's USB port (e.g., "/dev/ttyUSB0" for Linux/Mac)
# BAUD_RATE = 115200  # Default baud rate for ESP32
# TIMEOUT = 1  # Timeout for serial read (in seconds)

# def main():
#     try:
#         # Establish a serial connection
#         with serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT) as esp:
#             print(f"Connected to ESP32 on {PORT} at {BAUD_RATE} baud.")

#             while True:
#                 # Send a command to the ESP32
#                 command = input("Enter command to send (or 'exit' to quit): ")
#                 if command.lower() == "exit":
#                     print("Exiting...")
#                     break

#                 # Send the command to ESP32
#                 esp.write(command.encode('utf-8'))
#                 esp.write(b'\n')  # Send newline to terminate the command
#                 print(f"Sent: {command}")

#                 # Wait and read the response from the ESP32
#                 time.sleep(0.1)  # Small delay to allow ESP32 to respond
#                 response = esp.read_all().decode('utf-8').strip()
#                 print(f"Received: {response}")

#     except serial.SerialException as e:
#         print(f"Error: {e}")
#     except KeyboardInterrupt:
#         print("\nProgram interrupted. Exiting...")

# if __name__ == "__main__":
#     main()