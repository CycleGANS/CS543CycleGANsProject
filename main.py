import tensorflow as tf
import Generator as gen
import simple_discriminator as dis



# Creating placeholder for images
# Change the image_shape value
X = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
Y = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
GofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
FofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
GofFofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])
FofGofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3])


""" We will have 2 generators: G and F
	G : X -> Y
	F : Y -> X

	and 2 Discriminators: DX and DY

	DX aims to distinguish between images from {x} & translated images {F(y)}
	DY aims to distinguish between images from {y} & translated images {G(x)}
"""

# Creating the generators and discriminator networks
GofX = gen.generator(X, scope='G')
FofY = gen.generator(Y, scope='F')
GofFofY = gen.generator(FofY, scope='G')
FofGofX = gen.generator(GofX, scope='F')

D_Xlogits = dis.build_gen_discriminator(X, scope = 'DX')
D_FofYlogits = dis.build_gen_discriminator(FofY, scope = 'DX')
D_Ylogits = dis.build_gen_discriminator(Y, scope = 'DY')
D_GofXlogits = dis.build_gen_discriminator(GofX, scope = 'DY')

# Setting up losses for generators and discriminators
""" adv_losses are adversary losses
	cyc_losses are cyclic losses
	real_losses are losses from real images
	fake_losses are from generated images
"""
# https://arxiv.org/pdf/1611.04076.pdf this paper states that using cross entropy as loss
# causes the gradient to vanish. To avoid this problem, least squares losses are used as suggested by the paper.

# Adversary and Cycle Losses for G
G_adv_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.ones_like(D_GofXlogits)))
G_cyc_loss = tf.reduce_mean(tf.abs(GofFofY-Y)) * G_cyc_loss_weight        # Put lambda for G cyclic loss here
G_tot_loss = G_adv_loss + G_cyc_loss

# Adversary and Cycle Losses for F
F_adv_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits,tf.ones_like(D_FofYlogits)))
F_cyc_loss = tf.reduce_mean(tf.abs(FofGofX-X)) * F_cyc_loss_weight        # Put lambda for F cyclic loss here
F_tot_loss = F_adv_loss + F_cyc_loss

# Total Losses for G and F
GF_tot_loss = G_tot_loss + F_tot_loss


# Losses for DX
DX_real_loss = tf.reduce_mean(tf.squared_difference(D_Xlogits, tf.ones_like(D_Xlogits)))
DX_fake_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits, tf.zeros_like(D_FofYlogits)))
DX_tot_loss = (DX_real_loss+DX_fake_loss)/2

#Losses for DY
DY_real_loss = tf.reduce_mean(tf.squared_difference(D_Ylogits, tf.ones_like(D_Ylogits)))
DY_fake_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.zeros_like(D_GofXlogits)))
DY_tot_loss = (DY_real_loss+DY_fake_loss)/2

# Optimization
# Getting all the variables that belong to the different networks
# I.e. The weights and biases in G, F, DX and DY
network_variables = tf.trainable_variables()					#This gets all the variables that will be initialized
GF_variables = [variables for variables in network_variables if 'G' in variables.name or 'F' in variables.name]
DX_variables = [variables for variables in network_variables if 'DX' in variables.name]
DY_variables = [variables for variables in network_variables if 'DY' in variables.name]

optimizer = tf.train.AdamOptimizer(learning_rate)    			#Put the learning rate here
GF_train_step = optimizer.minimize(GF_tot_loss, var_list = GF_variables)
DX_train_step = optimizer.minimize(DX_tot_loss, var_list = DX_variables)
DY_train_step = optimizer.minimize(DY_tot_loss, var_list = DY_variables)


# Summary for Tensor Board
GF_summary = tf.summary.scalar("GF_tot_loss", GF_tot_loss)
DX_summary = tf.summary.scalar("DX_tot_loss", DX_tot_loss)
DY_summary = tf.summary.scalar("DY_tot_loss", DY_tot_loss)


# Training
# Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_summary_writer = tf.summary.FileWriter( summmary_path + '/train', sess.graph)
test_summary_writer = tf.summary.FileWriter( summmary_path + '/test', sess.graph)


for i in range(epochs):
	for j in range(no_of_batches):
		X_batch = # Put the X batch data here
		Y_batch = # Put the Y batch data here

		GofXforDis, FofYforDis = sess.run([GofX, FofY], feed_dict={X: X_batch, Y: Y_batch})

		DX_output, DX_vis_summary = sess.run([DX_train_step, DX_summary], feed_dict={X; X_batch, FofY: FofYforDis})

		DY_output, DY_vis_summary = sess.run([DY_train_step, DY_summary], feed_dict={Y: Y_batch, GofX: GofXforDis})

		GF_output, GF_vis_summ = sess.run([GF_train_step, GF_summary], feed_dict={X: X_batch, Y:Y_batch})

		train_summary_writer.add_summary(DX_vis_summary, j)
		train_summary_writer.add_summary(DY_vis_summary, j)
		train_summary_writer.add_summary(GF_vis_summ, j)



