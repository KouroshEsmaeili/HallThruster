import math
from abc import ABC, abstractmethod


# --- Abstract Base Class ---
class CurrentController(ABC):
    @abstractmethod
    def apply(self, present_value, control_value, dt):
        pass


# --- NoController ---
class NoController(CurrentController):
    def apply(self, present_value, control_value, dt):
        return control_value


# --- PIDController ---
class PIDController(CurrentController):
    def __init__(self, target_value: float,
                 proportional_constant: float = 1.0,
                 integral_constant: float = 0.0,
                 derivative_constant: float = 0.0,
                 errors=(0.0, 0.0, 0.0),
                 smoothing_frequency: float = 0.0,
                 smoothed_value: float = 0.0):
        self.target_value = target_value
        self.proportional_constant = proportional_constant
        self.integral_constant = integral_constant
        self.derivative_constant = derivative_constant
        self.errors = errors  # (e(t), e(t-dt), e(t-2dt))
        self.smoothing_frequency = smoothing_frequency
        self.smoothed_value = smoothed_value

    def apply(self, present_value, control_value, dt):
        K_smooth = math.exp(-dt * self.smoothing_frequency)

        self.smoothed_value = K_smooth * present_value + (1 - K_smooth) * self.smoothed_value

        new_error = self.target_value - present_value

        self.errors = (new_error, self.errors[0], self.errors[1])

        P = self.proportional_constant * (self.errors[0] - self.errors[1])
        I = self.integral_constant * self.errors[0] * dt
        D = self.derivative_constant * (self.errors[0] - 2 * self.errors[1] + self.errors[2]) / dt

        return control_value + P + I + D

    def __repr__(self):
        return (f"PIDController(target_value={self.target_value}, "
                f"Kp={self.proportional_constant}, Ki={self.integral_constant}, Kd={self.derivative_constant}, "
                f"errors={self.errors}, smoothing_frequency={self.smoothing_frequency}, "
                f"smoothed_value={self.smoothed_value})")


# --- Top-Level apply_controller Function ---
def apply_controller(controller: CurrentController, present_value, control_value, dt):
    if hasattr(controller, "apply"):
        return controller.apply(present_value, control_value, dt)
    else:
        return control_value


# --- Example Usage ---
if __name__ == "__main__":
    pid = PIDController(target_value=10.0,
                        proportional_constant=2.0,
                        integral_constant=0.5,
                        derivative_constant=0.1,
                        smoothing_frequency=1.0)

    present_value = 8.0
    control_value = 0.0
    dt = 0.1  # time step
    new_control = apply_controller(pid, present_value, control_value, dt)
    print("PIDController after update:", pid)
    print("New control value:", new_control)

    no_ctrl = NoController()
    new_control_no = apply_controller(no_ctrl, present_value, control_value, dt)
    print("NoController new control value:", new_control_no)
