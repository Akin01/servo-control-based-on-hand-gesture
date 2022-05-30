import serial


class Arduino:
    def __init__(self, port: int, baudrate: str):
        self.port = port
        self.baudrate = baudrate

        # serial connect through UART
        self.arduino = serial.Serial(port=port, baudrate=baudrate)

    def parseData(self, delimiter: str = '#') -> tuple:
        received_data = self.arduino.readline().decode().split(delimiter)
        return tuple(received_data)

    def sendData(self, data: any):
        self.arduino.write(data)
