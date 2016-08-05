# work with keras 1.0.6, 5th Aug 2016, Keunwoo Choi
import keras 
from keras.models import Model
from keras.layers import Input, GRU, Dense
from keras.layers import RepeatVector, merge

def get_model_approach_1(maxlen_enc, maxlen_dec, loss_function):
	'''Build an rnn model for audio summarisation.
	This model is trained per song-album pair.

	Parameters
	----------
	maxlen_enc: integer, length of encoder RNN (max number of tracks in an album)
	maxlen_dec: integer, length of decoder RNN (max number of words of album descriptions)
	loss_function: string, e.g. 'mse', 'kld', 'binary_crossentropy' or whatever keras supports.

	return: it returns a keras model.

	Encoder
	----------
	Encoder one input, which is a list of track_feature.
		(track feature=concat(np.mean(word_embeddings), tag_pred)).
	Word embeddings are the w2v vectors) of the words in the metadata/descriptions of a track. 
	tag_pred is a (50-dim) vector from the audio.
	Word embeddings are averaged to summarise each track.
	The output of encoder is a single vector that is a summary of the song - aka a context vector

	Decoder
	----------
	Decoder takes two inputs: a context vector and album_seq[:-1].
	The context vector is repeated and concatenated to album_seq[:-1].
	The output is album_seq[1:].
	'''
	n_hidden_enc = 128
	n_hidden_dec = 64
	dim_w2v = 300 # this is just a common dimension length. 
	dim_tag = 50 # because my convnet is trained to predict 50 tags.

	feats_input = Input(shape=(maxlen_enc, dim_w2v+dim_tag), name='feats_input') # feature for track, which is 300+50 dim
	
	album_seq_input = Input(shape=(maxlen_dec, dim_w2v), name='album_seq_input') # album_seq[:-1], for training of decoder

	# input
	encoded = GRU(n_hidden_enc, return_sequences=True, input_shape=(maxlen_enc, dim_w2v+dim_tag))(feats_input)
	encoded = GRU(n_hidden_enc, return_sequences=False)(encoded)
	# context vector
	context = RepeatVector(maxlen_dec)(encoded)

	# decoder
	x_decoder = merge([album_seq_input, context], mode='concat', concat_axis=2)
	
	decoded = GRU(n_hidden_dec, return_sequences=True, input_shape=(maxlen_dec, n_hidden_enc))(x_decoder)
	decoded = GRU(dim_w2v, return_sequences=True)(decoded)
	
	# model
	model = Model(input=[feats_input, album_seq_input], output=[decoded])
	optimiser = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(optimizer=optimiser, loss=loss_function)

	return model

