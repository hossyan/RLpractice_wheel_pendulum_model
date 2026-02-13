import numpy as np

class PID_controller:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self. pre_error = 0.0

    def calc(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        deriv = (error - self.pre_error) / dt

        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * deriv)
        self.pre_error = error

        u = np.clip(u, -1.0, 1.0)

        return u
    
    def reset(self):
        self.integral = 0.0
        self.pre_error = 0.0