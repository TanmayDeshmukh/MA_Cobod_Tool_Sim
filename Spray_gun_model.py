import numpy as np
class SprayGunModel:
    def __init__(self, beta1=1.5, beta2=4, a=1, b=2, f_max=1):
        self.beta1 = beta1
        self.beta2 = beta2
        self.a = a
        self.b = b
        self.f_max = f_max

    def deposition_intensity(self, x, y):
        # x_r = x * np.cos(orientation) - y * np.sin(orientation)
        # y_r = x * np.sin(orientation) + y * np.cos(orientation)
        # print(x**2, y**2,  )
        if abs(y) <= (self.b * np.sqrt(1 - (x ** 2) / self.a ** 2)):
            return self.f_max * pow(1.0 - (x ** 2) / self.a ** 2, self.beta1 - 1) * pow(
                1.0 - (y ** 2) / ((self.b ** 2) * (1 - (x ** 2) / self.a ** 2)), self.beta2 - 1)
        # print('nope')
        return 0.0
