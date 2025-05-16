import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

np.random.seed(42)  # my lucky number

### Part 1: Univariate Regression with Funky Functions ###
def funky1(x): return x * np.sin(x) + 2 * x
def funky2(x): return 10 * np.sin(x) + x**2
def funky3(x): return np.sign(x)*(x**2 + 300) + 20 * np.sin(x)

def inject_noise(y):
    return y + np.random.normal(0, 50, size=y.size) 

def do_funky_regression(funk, label, noisy=False, extra_features=False):
    print(f"\n\n=== Starting regression for {label} ===")
    n = 100
    stuff = np.linspace(-20, 20, n)
    res = funk(stuff)
    if noisy:
        res = inject_noise(res)

    featX = stuff.reshape(-1,1)
    if extra_features:
        featX = np.hstack([featX, stuff.reshape(-1,1)**2, np.sin(stuff).reshape(-1,1)])

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(featX, res, train_size=0.7, random_state=42, shuffle=True)

    models = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.2)),
        ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(20,), max_iter=1500, random_state=1)),
        ("RandomForest", RandomForestRegressor(n_estimators=25, random_state=1))
    ]

    for tag, model in models:
        model.fit(Xtrain, Ytrain)
        yguess = model.predict(Xtest)
        erry = mean_squared_error(Ytest, yguess)
        wow = r2_score(Ytest, yguess)
        print(f"-->{tag}: MSE = {erry:.2f} | R2 = {wow:.2f}")

    plt.figure()
    plt.scatter(Xtest[:,0], Ytest, label='Truth', color='red')
    plt.scatter(Xtest[:,0], model.predict(Xtest), label='Last model', alpha=0.7)
    plt.title(f"Wiggly Function: {label}{' w/Noise' if noisy else ''}{' +Features' if extra_features else ''}")
    plt.legend()
    plt.show()


do_funky_regression(funky1, 'funky1')
do_funky_regression(funky1, 'funky1', noisy=True)
do_funky_regression(funky1, 'funky1', extra_features=True)

do_funky_regression(funky2, 'funky2')
do_funky_regression(funky2, 'funky2', noisy=True)
do_funky_regression(funky2, 'funky2', extra_features=True)

do_funky_regression(funky3, 'funky3')
do_funky_regression(funky3, 'funky3', noisy=True)
do_funky_regression(funky3, 'funky3', extra_features=True)


### Part 2: Multistuff Regression ###
print("\n\n=== MULTI FEATURE MADNESS ===")
xX, yy = make_regression(n_samples=2000, n_features=10, n_informative=5, noise=8.7, random_state=42)
X1, X2, Y1, Y2 = train_test_split(xX, yy, train_size=0.75)

silly_mod = LinearRegression()
silly_mod.fit(X1, Y1)
predicto = silly_mod.predict(X2)

print("Multifeature MSE:", mean_squared_error(Y2, predicto))
print("R squared goes brr:", r2_score(Y2, predicto))
print("Fake feature weights:", silly_mod.coef_)


### Part 3: War Temp Time Machine ###
print("\n\n=== WAR TIME WEATHER ===")
try:
    data = pd.read_csv("/content/Summary of Weather.csv") # this path is used because i did it in colab, just Summary of Weather.csv can be used on different platforms 
    data["Date"] = pd.to_datetime(data["Date"], errors='coerce')
    data = data[data["STA"] == 22508][["Date", "MeanTemp"]].dropna()
    data = data.sort_values("Date")

    W = 7
    temps = data["MeanTemp"].values
    rolling_X = []
    rolling_y = []

    for i in range(len(temps) - W):
        rolling_X.append(temps[i:i+W])
        rolling_y.append(temps[i+W])

    rolling_X, rolling_y = np.array(rolling_X), np.array(rolling_y)
    datey = data["Date"].values[W:]

    X_train_roll = rolling_X[datey < np.datetime64("1945-01-01")]
    y_train_roll = rolling_y[datey < np.datetime64("1945-01-01")]
    X_test_roll = rolling_X[datey >= np.datetime64("1945-01-01")]
    y_test_roll = rolling_y[datey >= np.datetime64("1945-01-01")]

    bigbrain = RandomForestRegressor(n_estimators=100, random_state=99)
    bigbrain.fit(X_train_roll, y_train_roll)
    y_forecast = bigbrain.predict(X_test_roll)

    plt.figure(figsize=(10, 5))
    plt.plot(datey[datey >= np.datetime64("1945-01-01")], y_test_roll, label='Actual 1945 Temps', color='blue')
    plt.plot(datey[datey >= np.datetime64("1945-01-01")], y_forecast, label='Predicted Temps', color='orange')
    plt.title("HONOLULU ISLAND HEAT TIME TRAVEL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Forecast R2:", r2_score(y_test_roll, y_forecast))
    print("Forecast MSE:", mean_squared_error(y_test_roll, y_forecast))

    print("\nRolling with TimeSeriesSplit")
    splitter = TimeSeriesSplit(n_splits=5)
    for foldy, (a, b) in enumerate(splitter.split(rolling_X)):
        mod = LinearRegression()
        mod.fit(rolling_X[a], rolling_y[a])
        pred = mod.predict(rolling_X[b])
        print(f"Fold {foldy} R2: {r2_score(rolling_y[b], pred):.3f} | MSE: {mean_squared_error(rolling_y[b], pred):.3f}")
except Exception as e:
    print("Could not run temperature forecast section. Reason:", e)
