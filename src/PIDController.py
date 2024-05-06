'''
PID controller class

Author: Vrushabh Zinage

'''


class PIDController:

    def __init__(self, Kp, Ki, Kd, set_point=0):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.prev_error = 0
        self.integral = 0

    def compute(self, current_value):

        error = self.set_point - current_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return output

