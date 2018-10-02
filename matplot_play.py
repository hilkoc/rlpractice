import matplotlib.pyplot as plt
import math

def g(x):
    return (x - 2) * (x - 5) * (x-8)


x_values = list(range(1,11))
f_x = [-20 + x * x for x in x_values]
g_x = [g(x) for x in x_values]

h_x = [-0.5 * x + 3 for x in x_values]

pairs = dict()
for x, y in enumerate(h_x):
    pairs[x - 5] = y +  0.2 * (x-5)**2

pairs2 = dict()
for x,y in enumerate(h_x):
    pairs2[x+10] = y

print(pairs2)

def plot_it():
    plt.figure()
    #for i in range(0, len(steps)):
    plt.plot(x_values, f_x, label='f(x)')
    plt.plot(x_values, g_x, label='g(x)')

    xv, yv = map_to_plot(pairs)
    plt.plot(xv, yv, label='h(x)')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

def map_to_plot(map):
    sorted_keys = sorted(map.keys())
    xv, yv = sorted_keys, [map[k] for k in sorted_keys]
    return xv, yv



def plot_it():
    plt.figure()
    plt.scatter([2, 4, 6, 8], [11, 10, 6, 8], label='ordered')
    plt.scatter([2, 8, 6, 4], [10, 7, 5, 9], label='not ordered')

    plt.plot(*map_to_plot(pairs), label='pairs')
    plt.plot(*map_to_plot(pairs2), label='p2')
    x = [2, 8, 6, 4]  # data_dict.keys()
    y = [10,7, 5, 9]  # data_dict.values()
    # plt.scatter(x, y, label='scatter plt')
    plt.legend()
    plt.show()

def plot_planet_values():
    c1, c2 = 0, 0.3
    def g(c, x):
        v = 1 / (x-c)**2
        return v

    def n(c, x):
        v = math.exp(-5 * (x-c)**2)
        return v

    f = g

    x_values = [ -0.95 + i/10.0 for i in  range(0, 30)]
    f1_x = [f(c1,x) for x in x_values]
    f2_x = [f(c2, x) for x in x_values]
    fsum = [ a+b for a,b in zip(f1_x, f2_x)]

    plt.figure()
    #for i in range(0, len(steps)):
    plt.plot(x_values, f1_x, label='f(0,x)')
    plt.plot(x_values, f2_x, label='f(10,x)')
    plt.plot(x_values, fsum, label='som')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()


plot_planet_values()