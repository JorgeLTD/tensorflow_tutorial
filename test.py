import tensorflow as tf

# Nodos constantes
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

# Suma de nodos
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

# Placeholder de variables
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# modelo linear
# W y b son las variable a optimizar
# W=2 y b=4 son los objetivos de este ejemplo
W = tf.Variable([2.8], tf.float32)
b = tf.Variable([3.4], tf.float32)
x = tf.placeholder(tf.float32)
# grafo con el modelo
linear_model = W * x + b

# Constantes son inicializadas en declaracion
# Variables no son incializadas en declaración, de inicializan explicitamente
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# Para mejorar el modelo linear se necesita una referencia del valor esperado para la variable x
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# reduce_sum suma los valores de los errores a un solo valor escalar
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [6, 8, 10, 12]}))

# Optimización mediante descenso de gradiente
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(squared_deltas)

sess.run(init)  # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [6, 8, 10, 12]})
    print(sess.run([W, b]))
