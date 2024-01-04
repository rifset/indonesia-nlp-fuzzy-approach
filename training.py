from sklearn.model_selection import train_test_split

X = df_train.drop('bad_rating', axis=1)
Y = df_train.bad_rating
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=69)

target_classes = ['no', 'yes']

classes = np.unique(Y_train)
mapping = dict(zip(classes, target_classes))

classes, mapping

# no preprocessing
X_train_text_np = X_train.content
X_test_text_np = X_test.content

# standard preprocessing
from nlp_id.lemmatizer import Lemmatizer

lemmatizer = Lemmatizer()

X_train_text_sp = X_train.content.apply(preprocess).apply(
    lambda x: lemmatizer.lemmatize(x))
X_test_text_sp = X_test.content.apply(preprocess).apply(
    lambda x: lemmatizer.lemmatize(x))

# proposed preprocessing
X_train_text_me = X_train.content.apply(preprocess).apply(word_correction).apply(
    lambda x: lemmatizer.lemmatize(x))
X_test_text_me = X_test.content.apply(preprocess).apply(word_correction).apply(
    lambda x: lemmatizer.lemmatize(x))

# feature extraction + GloVe
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_tokens = 30

tokenizer_np = Tokenizer()
tokenizer_np.fit_on_texts(X_train_text_np)
X_train_vect_np = pad_sequences(tokenizer_np.texts_to_sequences(X_train_text_np),
                             maxlen=max_tokens,
                             padding="post",
                             truncating="post",
                             value=0.)
X_test_vect_np = pad_sequences(tokenizer_np.texts_to_sequences(X_test_text_np),
                            maxlen=max_tokens,
                            padding="post",
                            truncating="post",
                            value=0.)

tokenizer_sp = Tokenizer()
tokenizer_sp.fit_on_texts(X_train_text_sp)
X_train_vect_sp = pad_sequences(tokenizer_sp.texts_to_sequences(X_train_text_sp),
                             maxlen=max_tokens,
                             padding="post",
                             truncating="post",
                             value=0.)
X_test_vect_sp = pad_sequences(tokenizer_sp.texts_to_sequences(X_test_text_sp),
                            maxlen=max_tokens,
                            padding="post",
                            truncating="post",
                            value=0.)

tokenizer_me = Tokenizer()
tokenizer_me.fit_on_texts(X_train_text_me)
X_train_vect_me = pad_sequences(tokenizer_me.texts_to_sequences(X_train_text_me),
                             maxlen=max_tokens,
                             padding="post",
                             truncating="post",
                             value=0.)
X_test_vect_me = pad_sequences(tokenizer_me.texts_to_sequences(X_test_text_me),
                            maxlen=max_tokens,
                            padding="post",
                            truncating="post",
                            value=0.)

word_embeddings_np = np.zeros((len(tokenizer_np.index_word)+1, embed_len))
word_embeddings_sp = np.zeros((len(tokenizer_sp.index_word)+1, embed_len))
word_embeddings_me = np.zeros((len(tokenizer_me.index_word)+1, embed_len))

for idx, word in tokenizer_np.index_word.items():
    word_embeddings_np[idx] = glove_embeddings.get(word, np.zeros(embed_len))
    
for idx, word in tokenizer_sp.index_word.items():
    word_embeddings_sp[idx] = glove_embeddings.get(word, np.zeros(embed_len))
    
for idx, word in tokenizer_me.index_word.items():
    word_embeddings_me[idx] = glove_embeddings.get(word, np.zeros(embed_len))


# model learning
from tensorflow import reduce_sum
from keras.models import Model
from keras.layers import Dense, Embedding, Input

inputs = Input(shape=(max_tokens, ))
embeddings_layer = Embedding(input_dim=len(tokenizer_np.index_word) + 1,
                             output_dim=embed_len,
                             input_length=max_tokens,
                             trainable=False,
                             weights=[word_embeddings_np])
dense1 = Dense(128, activation="relu")
dense2 = Dense(64, activation="relu")
dense3 = Dense(len(target_classes), activation="softmax")

x = embeddings_layer(inputs)
x = reduce_sum(x, axis=1)
x = dense1(x)
x = dense2(x)
outputs = dense3(x)
model_np = Model(inputs=inputs, outputs=outputs)

model_np.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
model_np.fit(X_train_vect_np, Y_train, batch_size=32, epochs=8,
             validation_data=(X_test_vect_np, Y_test))

# evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_preds = model_np.predict(X_test_vect_np).argmax(axis=-1)

print('Using validation data')
print("Test Accuracy : {}".format(accuracy_score(Y_test, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_test, Y_preds, target_names=target_classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_test, Y_preds))
