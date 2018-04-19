import tensorflow as tf
import utils
import Generator as gen
import simple_discriminator as dis



# Creating placeholder for images
# Change the image_shape value
X = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3)
Y = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3)
#GofX = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3)
#FofY = tf.placeholder(tf.float32, [None, image_shape, image_shape, 3)


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

# Losses for G
G_adv_loss = tf.reduce_mean(tf.squared_difference(D_GofXlogits, tf.ones_like(D_GofXlogits)))
G_cyc_loss = tf.reduce_mean(tf.abs(GofFofY-Y))
G_tot_loss = G_adv_loss + G_cyc_loss_weight * G_cyc_loss       # Put lambda for G cyclic loss here

#Losses for F
F_adv_loss = tf.reduce_mean(tf.squared_difference(D_FofYlogits,tf.ones_like(D_FofYlogits)))
F_cyc_loss = tf.reduce_mean(tf.abs(FofGofX-X))
F_tot_loss = F_adv_loss + F_cyc_loss_weight * F_cyc_loss       # Put lambda for F cyclic loss here

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
G_variables = [variables for variables in network_variables if 'G' in variables.name]
F_variables = [variables for variables in network_variables if 'F' in variables.name]
DX_variables = [variables for variables in network_variables if 'DX' in variables.name]
DY_variables = [variables for variables in network_variables if 'DY' in variables.name]

optimizer = tf.train.AdamOptimizer(learning_rate)    			#Put the learning rate here
G_train_step = optimizer.minimize(G_tot_loss, var_list = G_variables)
F_train_step = optimizer.minimize(F_tot_loss, var_list = F_variables)
DX_train_step = optimizer.minimize(DX_tot_loss, var_list = DX_variables)
DY_train_step = optimizer.minimize(DY_tot_loss, var_list = DY_variables)


# Summary for Tensor Board
G_summary = utils.summary({G_adv_loss: 'G_adv_loss',
                           G_cyc_loss: 'G_cyc_loss',
                           G_tot_loss: 'G_tot_loss'})
F_summary = utils.summary({F_adv_loss: 'F_adv_loss',
                           F_cyc_loss: 'F_cyc_loss',
                           F_tot_loss: 'F_tot_loss'})
DX_summary = utils.summary({DX_real_loss: 'DX_real_loss',
							DX_fake_loss: 'DX_fake_loss',
							DX_tot_loss: 'DX_tot_loss'})
DY_summary = utils.summary({DY_real_loss: 'DY_real_loss',
							DY_fake_loss: 'DY_fake_loss',
							DY_tot_loss: 'DY_tot_loss'})