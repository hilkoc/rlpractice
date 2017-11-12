import matplotlib.pyplot as plt

def g(x):
    return (x - 2) * (x - 5) * (x-8)


x_values = list(range(1,11))
f_x = [-20 + x * x for x in x_values]
g_x = [g(x) for x in x_values]

h_x = [-0.5 * x + 3 for x in x_values]

pairs = dict()
for x,y in enumerate(h_x):
    pairs[x+100] = y

print(pairs)

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
    xv, yv = map.keys(), map.values()
    return xv, yv


xv, yv = map_to_plot(pairs)
print(type(xv))
print(type(yv))

# plot_it()
plt.figure()
plt.plot(*map_to_plot(pairs), label='the pairs')
plt.show()