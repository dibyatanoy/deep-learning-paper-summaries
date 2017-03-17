This is a list of summaries of papers in the field of deep learning I have been reading recently. The summaries are meant to help me internalize and keep a record of the techniques I have read about, and not as full analyses of the papers.

## **Speech Recognition with RNNs**

Paper: http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf

This paper deals with applying deep RNNs end-to-end for speech recognition - transcribing a sequence of acoustic data into a sequence of phonemes. The deep RNN architecture consists of many layers both across time and space. One major issue in speech recognition is aligning the acoustic input with the phoneme outputs, and the paper shows how to handle this using [CTCs](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf) or RNN transducers.

The architecture consists of LSTM (Long Short-Term Memory) cells and is bidirectional. A bidirectional RNN simply consists of two separate RNNs running in opposite directions along the sequence, and the output for each time step is a weighted average of the outputs from the two directions. The network is also deep, meaning that at every time step, there are hidden layers of LSTMs, and at each time step, the input for each hidden layer comes from the output from the previous layer (in case of the first hidden layer, the input is simply the $X$ values), as well as the output from the previous time step at the same layer.

An issue that remains is the aligning of input to output data- the input data is not segmented by hand. A CTC is added to the output layer of the RNN, and it at every time step, it emits softmax probabilities of $(K+1)$ symbols, where $K$ is the number of phonemes, and the 1 comes from a special blank $\phi$  symbol. Note that the CTC model does not look at the outputs from the previous time step - it only uses the output of the last hidden layer for the current time step. The probability of an output sequence is then the sum over all alignments that are equivalent to this sequence. For example, "He is" in the audio data can be transcribed as "[hi] _ [Iz]" or "_ _ [hiz]" (blanks denoting spaces), and both should be correct. This can be computed by using a variant of the Forward-Backward algorithm for HMMs (described [here](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf)).

** Note: ** An important point I realized later is that CTCs are applicable only when alignments are guaranteed to be monotonic. This means crossing alignments, such as $(1 2)$ and $(a b)$, with $a$ corresponding to $2$ and $b$ corresponding to $1$, cannot be represented.

The RNN transducer method seems to augment CTCs by adding a separate RNN at the prediction layer, so that the prediction at every time step can also depend on the predictions made so far. So for a label $k$, it obtains both the $Pr(k|t)$ from the CTC network, and $Pr(k|u)$ from the prediction network, These two values are multiplied together and normalized.

Points to ponder:
* What is meant by the layer size in the paper? Are there multiple LSTMs in each "layer"? Looks like yes, but here, the layer size seems to mean the size of the output from each layer. For example, for the first layer, if the input $x_t$ of dimenson $a$ is multiplied by a weight vector $W_t$ of dimensions $b \times a$, the layer size would be $b$.
* Will CTCs work for all alignment problems? Apparently no - they only seem to work when it is known that the length of the input sequence will NOT be less than that of the output sequence. This is obviously true for speech recognition.



### **RNN Encoder-Decoders in Statistical Machine Translation**

Paper: https://arxiv.org/pdf/1406.1078.pdf

In this paper, the authors describe an RNN based approach for encoding a variable length sequence into a fixed size vector (encoder), and then decoding a variable-length sequence from a fixed size vector. They claim the RNN encoder-decoder learns a continuous space representation of phrases that preserve both semantic and syntactic content.

The output of the encoder is the hidden state $c$ of the RNN at the last time step. For the decoder RNN, the hidden state at every time step is a function of the previous hidden state, the previous output, and $c$. That is, $h_{t} = f(h_{t-1}, y_{t-1}, c)$. The two components are then jointly trained to maximize the conditional normalized sum of the log likelihoods (the log likelihood of a single sequence-to-sequence translation is $\log{p_{\theta}(y_n | x_n)}$).

The authors also introduce a new hidden unit similar to the LSTM. This consists of a reset gate $r_j$ and an update gate $z_j$. The reset gate decides how much of the previous hidden state to include in computing a temporary new hidden state $\tilde{h_t}$, which also depends on the input $x_t$, while the update gate decides how much information from the previous hidden state will carry over to the current hidden state. So: $$h_j^t = z_jh_j^{t-1} + (1-z_j)\tilde{h^t_j}$$

The RNN encoder decoder is applied in the scoring of phrase pairs in language translation. The statistical model of translation tries to find $f$ that maximizes $p(f|e) = p(e|f)p(e)$ (the translation and language model terms, respectively), given an input $e$. Phrase pairs from the two languages can be fed into the system, and the score is simply $p_{\theta}(y|x)$, where $(x, y)$ is the phrase pair. This score can then add as an additional feature in the model.

The authors also mention the use of [CSLM](https://ufal.mff.cuni.cz/pbml/93/art-schwenk.pdf) in their models, which uses NNs for the language model. It appears that the contributions of the RNN encoder-decoder and CSLM are independent. The authors claim that the embeddings generated by the encoder also capture both syntactic and semantic content of the phrases.



### **RNN Encoder-Decoders in Statistical Machine Translation**

Paper: https://arxiv.org/pdf/1406.1078.pdf

In this paper, the authors describe an RNN based approach for encoding a variable length sequence into a fixed size vector (encoder), and then decoding a variable-length sequence from a fixed size vector. They claim the RNN encoder-decoder learns a continuous space representation of phrases that preserve both semantic and syntactic content.

The output of the encoder is the hidden state $c$ of the RNN at the last time step. For the decoder RNN, the hidden state at every time step is a function of the previous hidden state, the previous output, and $c$. That is, $h_{t} = f(h_{t-1}, y_{t-1}, c)$. The two components are then jointly trained to maximize the conditional normalized sum of the log likelihoods (the log likelihood of a single sequence-to-sequence translation is $\log{p_{\theta}(y_n | x_n)}$).

The authors also introduce a new hidden unit similar to the LSTM. This consists of a reset gate $r_j$ and an update gate $z_j$. The reset gate decides how much of the previous hidden state to include in computing a temporary new hidden state $\tilde{h_t}$, which also depends on the input $x_t$, while the update gate decides how much information from the previous hidden state will carry over to the current hidden state. So: $$h_j^t = z_jh_j^{t-1} + (1-z_j)\tilde{h^t_j}$$

The RNN encoder decoder is applied in the scoring of phrase pairs in language translation. The statistical model of translation tries to find $f$ that maximizes $p(f|e) = p(e|f)p(e)$ (the translation and language model terms, respectively), given an input $e$. Phrase pairs from the two languages can be fed into the system, and the score is simply $p_{\theta}(y|x)$, where $(x, y)$ is the phrase pair. This score can then add as an additional feature in the model.

The authors also mention the use of [CSLM](https://ufal.mff.cuni.cz/pbml/93/art-schwenk.pdf) in their models, which uses NNs for the language model. It appears that the contributions of the RNN encoder-decoder and CSLM are independent. The authors claim that the embeddings generated by the encoder also capture both syntactic and semantic content of the phrases.
