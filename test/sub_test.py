import time
import roslibpy

ROS_HOST = "localhost"  # ROS2 서버 IP로 변경
ROS_PORT = 9090
TOPIC_NAME = "/arduino_data"
TOPIC_TYPE = "std_msgs/Float32MultiArray"


"""
target_pressure,current_pressure,filtered_pressure,valve
"""


class RosSubscriber:
    def __init__(self, host: str, port: int, topic_name: str, topic_type: str):
        self.client = roslibpy.Ros(host=host, port=port)
        self.topic = roslibpy.Topic(self.client, topic_name, topic_type)

    def callback(self, message):
        print(f"[RECV] {TOPIC_NAME}: {message['data']}")

    def start(self):
        self.client.run()

        if not self.client.is_connected:
            raise RuntimeError("rosbridge connection failed")

        print(f"Connected to rosbridge: ws://{ROS_HOST}:{ROS_PORT}/")

        self.topic.subscribe(self.callback)

    def __delete__(self, instance):
        self.topic.unsubscribe()
        self.client.terminate()


def main():
    subscriber = RosSubscriber(ROS_HOST, ROS_PORT, TOPIC_NAME, TOPIC_TYPE)
    subscriber.start()

    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
