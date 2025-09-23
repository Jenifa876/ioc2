import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.client = self.create_client(AddTwoInts, 'add_t>
        while not self.client.wait_for_service(timeout_sec=>            self.get_logger().info('Service not available, >
    def send_request(self, a, b):
        req = AddTwoInts.Request()
        req.a = a
        req.b = b
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    node = AddTwoIntsClient()
    response = node.send_request(7, 5)
    node.get_logger().info(f'Result of 7 + 5 = {response.su>    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
