import tensorflow as tf

a = tf.constant(2, name='A')
x = tf.Variable(0, name='X')
y = tf.multiply(a,x, name='Y')

x_inc = tf.assign(x, x+1, name='X_INC')

sess = tf.InteractiveSession()

tf.summary.scalar('A_SCALAR', a)
tf.summary.scalar('X_SCALAR', x)
tf.summary.scalar('Y_SCALAR', y)
merged = tf.summary.merge_all()

if tf.gfile.Exists('./logdir'):
    tf.gfile.DeleteRecursively('./logdir')
writer = tf.summary.FileWriter('./logdir', sess.graph)


init = tf.global_variables_initializer()
sess.run(init)

for i in range(3):
    summary, result = sess.run([merged, y])
    writer.add_summary(summary, i)
    print(result)
    sess.run(x_inc)

sess.close()
