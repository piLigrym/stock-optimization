import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Вхідні змінні
price = ctrl.Antecedent(np.arange(0, 101, 1), 'price')
volume = ctrl.Antecedent(np.arange(0, 1001, 1), 'volume')
market_stability = ctrl.Antecedent(np.arange(0, 101, 1), 'market_stability')

# Вихідна змінна
recommended_share = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'recommended_share')

# Функції належності для price
price['low'] = fuzz.trapmf(price.universe, [0, 0, 20, 40])
price['medium'] = fuzz.trimf(price.universe, [20, 50, 80])
price['high'] = fuzz.trapmf(price.universe, [60, 80, 100, 100])

# Функції належності для volume
volume['low'] = fuzz.trapmf(volume.universe, [0, 0, 200, 400])
volume['medium'] = fuzz.trimf(volume.universe, [200, 600, 1000])
volume['high'] = fuzz.trapmf(volume.universe, [800, 1000, 1000, 1000])

# Функції належності для market_stability
market_stability['unstable'] = fuzz.trapmf(market_stability.universe, [0, 0, 20, 40])
market_stability['average'] = fuzz.trimf(market_stability.universe, [20, 50, 80])
market_stability['stable'] = fuzz.trapmf(market_stability.universe, [60, 80, 100, 100])

# Функції належності для recommended_share
recommended_share['low'] = fuzz.trimf(recommended_share.universe, [0, 0, 0.5])
recommended_share['medium'] = fuzz.trimf(recommended_share.universe, [0, 0.5, 1])
recommended_share['high'] = fuzz.trimf(recommended_share.universe, [0.5, 1, 1])

# Правила
rules = [
    ctrl.Rule(price['low'] & volume['low'] & market_stability['unstable'], recommended_share['low']),
    ctrl.Rule(price['low'] & volume['low'] & market_stability['average'], recommended_share['low']),
    ctrl.Rule(price['low'] & volume['low'] & market_stability['stable'], recommended_share['low']),
    ctrl.Rule(price['low'] & volume['medium'] & market_stability['unstable'], recommended_share['low']),
    ctrl.Rule(price['low'] & volume['medium'] & market_stability['average'], recommended_share['medium']),
    ctrl.Rule(price['low'] & volume['medium'] & market_stability['stable'], recommended_share['medium']),
    ctrl.Rule(price['low'] & volume['high'] & market_stability['unstable'], recommended_share['low']),
    ctrl.Rule(price['low'] & volume['high'] & market_stability['average'], recommended_share['medium']),
    ctrl.Rule(price['low'] & volume['high'] & market_stability['stable'], recommended_share['high']),
    ctrl.Rule(price['medium'] & volume['low'] & market_stability['unstable'], recommended_share['low']),
    ctrl.Rule(price['medium'] & volume['low'] & market_stability['average'], recommended_share['low']),
    ctrl.Rule(price['medium'] & volume['low'] & market_stability['stable'], recommended_share['medium']),
    ctrl.Rule(price['medium'] & volume['medium'] & market_stability['unstable'], recommended_share['low']),
    ctrl.Rule(price['medium'] & volume['medium'] & market_stability['average'], recommended_share['medium']),
    ctrl.Rule(price['medium'] & volume['medium'] & market_stability['stable'], recommended_share['high']),
    ctrl.Rule(price['medium'] & volume['high'] & market_stability['unstable'], recommended_share['medium']),
    ctrl.Rule(price['medium'] & volume['high'] & market_stability['average'], recommended_share['high']),
    ctrl.Rule(price['medium'] & volume['high'] & market_stability['stable'], recommended_share['high']),
    ctrl.Rule(price['high'] & volume['low'] & market_stability['unstable'], recommended_share['low']),
    ctrl.Rule(price['high'] & volume['low'] & market_stability['average'], recommended_share['medium']),
    ctrl.Rule(price['high'] & volume['low'] & market_stability['stable'], recommended_share['high']),
    ctrl.Rule(price['high'] & volume['medium'] & market_stability['unstable'], recommended_share['medium']),
    ctrl.Rule(price['high'] & volume['medium'] & market_stability['average'], recommended_share['high']),
    ctrl.Rule(price['high'] & volume['medium'] & market_stability['stable'], recommended_share['high']),
    ctrl.Rule(price['high'] & volume['high'] & market_stability['unstable'], recommended_share['medium']),
    ctrl.Rule(price['high'] & volume['high'] & market_stability['average'], recommended_share['high']),
    ctrl.Rule(price['high'] & volume['high'] & market_stability['stable'], recommended_share['high']),
]

# Створення системи контролю
share_ctrl = ctrl.ControlSystem(rules)
share_simulation = ctrl.ControlSystemSimulation(share_ctrl)

# Приклад використання
share_simulation.input['price'] = 70
share_simulation.input['volume'] = 900
share_simulation.input['market_stability'] = 80

share_simulation.compute()

print(f"Recommended Portfolio Share: {share_simulation.output['recommended_share']:.2f}")

# Моделювання поверхні за алгоритмом Сугено
x_price = np.arange(0, 101, 1)
y_volume = np.arange(0, 1001, 1)
X, Y = np.meshgrid(x_price, y_volume)
Z_share = np.zeros_like(X, dtype=float)

for i in range(len(x_price)):
    for j in range(len(y_volume)):
        share_simulation.input['price'] = x_price[i]
        share_simulation.input['volume'] = y_volume[j]
        share_simulation.input['market_stability'] = 80  # Фіксоване значення
        share_simulation.compute()
        Z_share[j, i] = share_simulation.output['recommended_share']

# Візуалізація поверхні в 3D стилі
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_share, cmap='viridis')

ax.set_title('Recommended Portfolio Share')
ax.set_xlabel('Price')
ax.set_ylabel('Volume')
ax.set_zlabel('Recommended Share')

plt.show()