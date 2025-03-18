# AttenNet
In this repository, I will implement a Transformer model, including attention layers, encoder-decoder architecture, and the full module, from scratch using PyTorch.


## Embedding Layer

The embedding layer is the first layer in the model. It converts the input tokens into dense vectors of fixed size. The embedding layer is a simple lookup table that stores embeddings of a fixed dictionary and size. The embedding layer is initialized with random weights and is updated during training.



## Positional Encoding

The positional encoding is added to the input embeddings to give the model information about the position of the tokens in the sequence. The positional encoding is a sine and cosine function of different frequencies.



## Attention Layer

The attention layer is the core building block of the Transformer model. It computes the attention scores between the query and key vectors and uses these scores to compute the weighted sum of the value vectors. The attention layer consists of three main components: query, key, and value matrices.

```maths
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

where Q, K, and V are the query, key, and value matrices, respectively, and d_k is the dimension of the key vectors.

## Tokenization

Tokenization is the process of converting text into tokens. In this implementation, I will use the `torchtext` library to tokenize the text data. I have created a custom tokenizer that tokenizes the text data and converts the tokens into integer indices using ['Byte-Pair Encoding'](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0) (BPE) algorithm. The BPE algorithm is a simple data compression technique that replaces the most frequent pair of bytes with a single byte. 