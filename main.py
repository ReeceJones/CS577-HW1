# general libraries
from typing import List, Tuple
import pandas as pd
import numpy as np

# nltk libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def preprocess(df : pd.DataFrame, schema=None, text_col='text', min_ngram=1, max_ngram=1, **kwargs):
    wnl = WordNetLemmatizer()

    def rebalance(df):
        # rebalance data so that each label has uniform probability of being encountered
        counts = df.emotions.value_counts().sort_values()
        a = pd.DataFrame(columns=df.columns)
        mc = 500
        for e in df.emotions.unique():
            a = pd.concat([a, df[df.emotions==e].sample(n=mc, replace=True, ignore_index=True)])

        return a.sample(frac=1).reset_index(drop=True)

    def tag_tokenize(x):
        if 'lemmatize' in kwargs and kwargs['lemmatize']==True:
            return pos_tag(word_tokenize(x.lower()))
        else:
            return word_tokenize(x.lower())

    def lemmatize(x):
        # https://stackoverflow.com/questions/35870282/nltk-lemmatizer-and-pos-tag
        wnpos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
        return [wnl.lemmatize(y[0], wnpos(y[1])) for y in x]

    stops = set(stopwords.words('english'))
    def remove_stopwords(x):
        return [y for y in x if y not in stops]

    def build_ngrams(x, l, u):
        r = x
        z = list()
        for i in range(len(r)):
            for j in range(l-1, u):
                y = r[max(i-j,0):min(i+1, len(r))]
                if len(y) == j+1:
                    z.append(' '.join(y))
        return z

    def build_tf(x, word):
        return x.count(word)

    def build_bow(x, ignore_cols):
        if x.name not in ignore_cols:
            return x.apply(lambda x: int(x > 0))
        else:
            return x

    def onehot_encode(x, classes):
        return [int(x==y) for y in classes]

    if 'rebalance' in kwargs and kwargs['rebalance']==True:
        a = rebalance(df)
    else:
        a = df.copy()
    # transform text
    a['text'] = a.text.apply(tag_tokenize)
    if 'lemmatize' in kwargs and kwargs['lemmatize']==True:
        a['text'] = a.text.apply(lemmatize)
    if 'remove_stopwords' in kwargs and kwargs['remove_stopwords']==True:
        a['text'] = a.text.apply(remove_stopwords)
    a['text'] = a.text.apply(build_ngrams, args=(min_ngram,max_ngram))

    # build schema
    if schema is None:
        schema = dict()
        for idx, row in a.iterrows():
            for w in row.text:
                if w in schema:
                    schema[w] = schema[w] + 1
                else:
                    schema[w] = 1

    sorted_labels = [x[0] for x in sorted([(k,v) for k,v in schema.items()], key=lambda x: x[1], reverse=True)]

    if 'feature_count' in kwargs:
        n = kwargs['feature_count']
        for w in sorted_labels[n:]:
            del schema[w]

    # apply corpus to text
    text = a.text
    del a['text']
    ignore_cols = list(a.columns)
    for w in schema.keys():
        a = pd.concat([a,pd.DataFrame({f'_{w}': text.apply(build_tf, word=w)})], axis=1)

    if 'use_tf' not in kwargs or kwargs['use_tf']==False:
        a = a.apply(build_bow, ignore_cols=ignore_cols)

    # onehot encode target column
    classes = list()
    if 'emotions' in a.columns:
        classes = sorted(a.emotions.unique().tolist())
        a['emotions'] = a.emotions.apply(onehot_encode, classes=classes)

    return a, schema, classes

def train_lr(df, train_lambda, train_step_size, train_max_iter, train_tolerance, ignore_cols=['id'], label_col='emotions', **kwargs) -> np.matrix:
    # https://cedar.buffalo.edu/~srihari/CSE574/Chap4/4.3.4-MultiLogistic.pdf
    # get vector, weight, and target matrices
    print(f'Training LR model using {train_lambda=}, {train_step_size=}, {train_max_iter=}, {train_tolerance=} over {df.shape[0]} rows and {df.shape[1]-len(ignore_cols)-1} features...')
    target_matrix = np.matrix([np.array(x) for x in df[label_col].tolist()])
    vector_matrix = np.matrix(df.loc[:, ~df.columns.isin(ignore_cols+[label_col])])
    # add bias term
    vector_matrix = np.insert(vector_matrix, vector_matrix.shape[1], np.ones(vector_matrix.shape[0]), axis=1)
    weight_matrix = np.matrix(np.zeros((target_matrix.shape[1], vector_matrix.shape[1])))

    i = 0
    old_weight_matrix = weight_matrix.copy()
    while i < train_max_iter and (np.linalg.norm(weight_matrix-old_weight_matrix) >= train_tolerance or i < 1):
        old_weight_matrix = weight_matrix.copy()
        i = i + 1
        # Can compute softmax using matrix multiplication:
        # (V)(W.T)
        dp = np.exp(vector_matrix @ weight_matrix.T)
        # then normalize each row in the resulting matrix and take difference of true label
        dp = (dp/dp.sum(axis=1)) - target_matrix
        # then compute gradient by adding regularization term and ((V.T)(L)).T
        tg = (weight_matrix * train_lambda) + (vector_matrix.T @ dp).T
        # apply gradient
        weight_matrix = weight_matrix - (train_step_size * tg)

    return weight_matrix

def apply_lr(df, model, ignore_cols=['id','emotions']) -> pd.DataFrame:
    a = df.copy()
    vector_matrix = np.matrix(df.loc[:, ~df.columns.isin(ignore_cols)])
    # add bias term
    vector_matrix = np.insert(vector_matrix, vector_matrix.shape[1], np.ones(vector_matrix.shape[0]), axis=1)

    # ultimately, it doesn't matter that we apply softmax here since each entry will only differ
    # by a multiplicative normalizing constant wrt to its row, but do it anyways for consistency
    dp = np.exp(vector_matrix @ model.T)
    a['predictions'] = (dp/dp.sum(axis=1)).tolist()

    return a


def LR():
    # learn a model and do any necessary processing
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    preprocessed_train_data, schema, classes = preprocess(train_data, min_ngram=1, max_ngram=3, lemmatize=False, remove_stopwords=True, feature_count=2000)
    preprocessed_test_data,_,__ = preprocess(test_data, schema=schema, min_ngram=1, max_ngram=3, lemmatize=False, remove_stopwords=True, feature_count=2000)
    model = train_lr(preprocessed_train_data, train_lambda=0.05, train_step_size=0.05, train_max_iter=500, train_tolerance=0.00001)
    transformed_data = apply_lr(preprocessed_test_data, model)
    test_data['emotions'] = transformed_data.predictions.apply(lambda x: classes[np.argmax(x)])
    test_data.to_csv('test_lg.csv', index=False)

def mat_sigmoid(W, X):
    return 1 / (1 + np.exp(-(W @ X)))

def mat_sigmoid_dv(W, X):
    dp = W @ X
    dp = 1/(1+np.exp(-dp))
    return np.multiply(dp, 1-dp)

def mat_swish1(W,X):
    dp = W @ X
    return dp / (1 + np.exp(-dp))

def mat_swish1_dv(W,X):
    dp = W @ X
    return (1 + np.exp(-dp) + np.multiply(dp, np.exp(-dp))) / np.power(1 + np.exp(-dp),2)

def mat_relu(W,X):
    dp = W @ X
    return np.multiply((dp > 0), dp)

def mat_relu_dv(W,X):
    dp = W @ X
    return (dp > 0) * 1

def mat_softmax(W,X):
    dp = W @ X
    z = np.exp(dp - np.max(dp))
    return z/z.sum(axis=0)

def mat_softmax_jc(W, X, da):
    # compute softmax values
    z = mat_softmax(W,X)
    # https://themaverickmeerkat.com/2019-10-23-Softmax/
    p = z.T
    m, n = p.shape
    t1 = np.einsum('ij,ik->ijk', p, p)
    t2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))
    dS = t2 - t1
    return np.einsum('ijk,ik->ij', dS, da.T).T

def mat_squared_loss_dv(A, G):
    return -(G-A)

def train_nn(df, layers, train_step_size, train_iter, dropout_rate, batch_size=0, ignore_cols=['id'], label_col='emotions', **kwargs):
    # https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication
    # get vector, weight, and target matrices
    print(f'Training NN model using {layers=}, {train_step_size=}, {train_iter=}, {dropout_rate=}, {batch_size=} over {df.shape[0]} rows and {df.shape[1]-len(ignore_cols)-1} features...')
    target_matrix = np.matrix([np.array(x) for x in df[label_col].tolist()]).T
    vector_matrix = np.matrix(df.loc[:, ~df.columns.isin(ignore_cols+[label_col])])
    # add bias term
    vector_matrix = np.insert(vector_matrix, vector_matrix.shape[1], np.ones(vector_matrix.shape[0]), axis=1).T

    # unpack layer data
    layer_sizes = [l[0] for l in layers]
    activations = [None] + [l[1] if len(l) == 3 else mat_sigmoid for l in layers] + [mat_softmax]
    activation_derivatives = [None] + [l[2] if len(l) == 3 else mat_sigmoid_dv for l in layers] + [mat_softmax_jc]

    # assume fully connected
    weight_shapes = list()
    if len(layer_sizes) == 0:
        weight_shapes = [(target_matrix.shape[0], vector_matrix.shape[0])]
    else:
        weight_shapes = [(layer_sizes[0] + 1, vector_matrix.shape[0])]
        for l in range(1, len(layer_sizes)):
            weight_shapes.append((layer_sizes[l] + 1, layer_sizes[l-1] + 1))
        weight_shapes.append((target_matrix.shape[0], layer_sizes[-1] + 1))
    # weights = [None, np.matrix(np.random.randn(6, vector_matrix.shape[1])), np.matrix(np.random.randn(target_matrix.shape[1], 6))]
    # print(weight_shapes)
    weights = [None] + [np.matrix(np.random.randn(*s)) for s in weight_shapes]

    # don't need to treat bias term super specially, just need to fix A value to 1, and derivative to 0?

    for i in range(1,train_iter+1):
        # forward feed phase
        # activations
        batch = np.random.choice(vector_matrix.shape[1], 75) if batch_size > 0 else None
        # A = [vector_matrix[:,batch]]
        A = [vector_matrix[:, batch] if batch_size > 0 else vector_matrix]
        labels = target_matrix[:, batch] if batch_size > 0 else target_matrix
        # activation derivatives
        D = [A[-1]]
        for l in range(1, len(weights)):
            Xp = np.copy(A[-1])
            # add bias term if needed
            # if l < len(weights)-1:
            Xp[Xp.shape[0]-1] = 1
            a = activations[l](weights[l], Xp)
            d = activation_derivatives[l](weights[l], Xp, a - labels) if l==len(weights)-1 else activation_derivatives[l](weights[l], Xp)
            # apply dropout
            if dropout_rate > 0 and l < len(weights)-1:
                dead_neurons = np.random.choice(a.shape[0], int(a.shape[0] * dropout_rate))
                a[dead_neurons] = 0
                d[dead_neurons] = 0
                a = a / (1 - dropout_rate)
                d = d / (1 - dropout_rate)
            A.append(a)
            D.append(d)

        # acc = np.sum(np.argmax(A[-1], axis=0) == np.argmax(target_matrix, axis=0)) / target_matrix.shape[1]
        # err = -np.sum(np.sum(np.multiply(target_matrix, np.nan_to_num(np.log(A[-1]))), axis=0)) # categorical cross entropy loss
        # print(i, acc, err)

        # back prop phase
        # partial product
        # P = np.multiply(D[-1], (A[-1] - target_matrix))
        # with softmax, we already include dC/da
        P = D[-1]
        # gradients
        # G = list()
        for l in range(len(A)-1, 0, -1):
            g = P @ A[l-1].T
            P = np.multiply(D[l-1], weights[l].T @ P)
            weights[l] = weights[l] - (train_step_size * g)

    return list(zip(weights, activations))
    
            
def apply_nn(df, model, ignore_cols=['id','emotions']) -> pd.DataFrame:
    b = df.copy()
    vector_matrix = np.matrix(df.loc[:, ~df.columns.isin(ignore_cols)])
    # add bias term
    vector_matrix = np.insert(vector_matrix, vector_matrix.shape[1], np.ones(vector_matrix.shape[0]), axis=1).T
    A = [vector_matrix]
    for l in range(1, len(model)):
        Xp = np.copy(A[-1])
        Xp[Xp.shape[0]-1] = 1
        a = model[l][1](model[l][0], Xp)
        A.append(a)
    
    b['predictions'] = A[-1].T.tolist()
    return b

def NN():
    # your Multi-layer Neural Network
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    del train_data['text']
    test_text = test_data.text
    del test_data['text']
    embeddings = pd.read_csv('naive_glove_embeddings.csv')
    train_data_embeddings = pd.merge(train_data, embeddings, on='id')
    test_data_embeddings = pd.merge(test_data, embeddings, on='id')
    classes = sorted(list(train_data.emotions.unique()))
    # one-hot encode label
    train_data_embeddings.emotions = train_data_embeddings.emotions.apply(lambda x: [int(x==y) for y in classes])
    model = train_nn(train_data_embeddings, layers=[(30, mat_sigmoid, mat_sigmoid_dv)], train_step_size=0.01, train_iter=20000, dropout_rate=0.15, batch_size=50)
    transformed_data = apply_nn(test_data_embeddings, model)
    test_data['text'] = test_text
    test_data['emotions'] = transformed_data.predictions.apply(lambda x: classes[np.argmax(x)])
    test_data.to_csv('test_nn.csv', index=False)


if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    print ("..................Beginning of Logistic Regression................")
    LR()
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    NN()
    print ("..................End of Neural Network................")