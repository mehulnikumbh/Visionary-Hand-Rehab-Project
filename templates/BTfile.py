import bluetooth


def discover_devices():
    nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True, lookup_class=False)

    print("Found {} devices:".format(len(nearby_devices)))
    for i, (addr, name) in enumerate(nearby_devices):
        print("  {}. {} - {}".format(i + 1, addr, name))
    return nearby_devices


def select_device(devices):
    while True:
        try:
            choice = int(input("Enter the number of the device you want to connect to: "))
            if 1 <= choice <= len(devices):
                return devices[choice - 1][0]
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def connect_to_device(device_address):
    port = 1  # Standard port for Bluetooth communication
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

    try:
        sock.connect((device_address, port))
        print("Connected to device:", device_address)
        return sock
    except Exception as e:
        print("Error:", e)
        return None


def send_command(sock, command):
    try:
        sock.send(command.encode())
        print("Command sent:", command)
    except Exception as e:
        print("Error:", e)


def close_connection(sock):
    try:
        sock.close()
        print("Connection closed.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    devices = discover_devices()
    if devices:
        device_address = select_device(devices)
        sock = connect_to_device(device_address)

        if sock:
            # Example: Sending a command "Hello" to the device
            send_command(sock, "Hello")

            # You can send more commands here

            close_connection(sock)
