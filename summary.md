<br>
<br>
<br>

This is a list of summaries of papers in the field of deep learning I have been reading recently. The summaries are meant to help me internalize and keep a record of the techniques I have read about, and not as full analyses of the papers.

# Table of contents
* [Speech Recognition With Deep Recurrent Neural Networks](#speech-recognition-with-rnns)
	* Graves, Mohamed and Hinton
* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](#rnn-encoder-decoders-in-statistical-machine-translation)
	* Cho, van Merrienboer, Gulcehre, Bahdanau, Bougares, Schwenk and Bengio
* [Sequence to Sequence Learning with Neural Networks](#sequence-to-sequence-learning-with-deep-rnns)
	* Sutskever, Vinyals and V. Le
* [Attention-based Neural Machine Translation](#attention-based-neural-machine-translation)
	* Luong, Pham and Manning (2015)

<br>

### **Speech Recognition with RNNs**

Paper: http://www.cs.toronto.edu/~fritz/absps/RNN13.pdf

This paper deals with applying deep RNNs end-to-end for speech recognition - transcribing a sequence of acoustic data into a sequence of phonemes. The deep RNN architecture consists of many layers both across time and space. One major issue in speech recognition is aligning the acoustic input with the phoneme outputs, and the paper shows how to handle this using [CTCs](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempo .pdf) or RNN transducers.

The architecture consists of LSTM (Long Short-Term Memory) cells and is bidirectional. A bidirectional RNN simply consists of two separate RNNs running in opposite directions along the sequence, and the output for each time step is a weighted average of the outputs from the two directions. The network is also deep, meaning that at every time step, there are hidden layers of LSTMs, and at each time step, the input for each hidden layer comes from the output from the previous layer (in case of the first hidden layer, the input is simply the $X$ values), as well as the output from the previous time step at the same layer.

An issue that remains is the aligning of input to output data- the input data is not segmented by hand. A CTC is added to the output layer of the RNN, and it at every time step, it emits softmax probabilities of $(K+1)$ symbols, where $K$ is the number of phonemes, and the 1 comes from a special blank $\phi$  symbol. Note that the CTC model does not look at the outputs from the previous time step - it only uses the output of the last hidden layer for the current time step. The probability of an output sequence is then the sum over all alignments that are equivalent to this sequence. For example, "He is" in the audio data can be transcribed as "[hi] _ [Iz]" or "_ _ [hiz]" (blanks denoting spaces), and both should be correct. This can be computed by using a variant of the Forward-Backward algorithm for HMMs (described [here](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf)).

* **Note**:  An important point I realized later is that CTCs are applicable only when alignments are guaranteed to be monotonic. This means crossing alignments, such as $(1 2)$ and $(a b)$, with $a$ corresponding to $2$ and $b$ corresponding to $1$, cannot be represented.

The RNN transducer method seems to augment CTCs by adding a separate RNN at the prediction layer, so that the prediction at every time step can also depend on the predictions made so far. So for a label $k$, it obtains both the $Pr(k|t)$ from the CTC network, and $Pr(k|u)$ from the prediction network, These two values are multiplied together and normalized.

Points to ponder

* What is meant by the layer size in the paper? Are there multiple LSTMs in each "layer"? Looks like yes, but here, the layer size seems to mean the size of the output from each layer. For example, for the first layer, if the input $x_t$ of dimenson $a$ is multiplied by a weight vector $W_t$ of dimensions $b \times a$, the layer size would be $b$.
* Will CTCs work for all alignment problems? Apparently no - they only seem to work when it is known that the length of the input sequence will NOT be less than that of the output sequence. This is obviously true for speech recognition.



### **RNN Encoder-Decoders in Statistical Machine Translation**

Paper: https://arxiv.org/pdf/1406.1078.pdf

In this paper, the authors describe an RNN based approach for encoding a variable length sequence into a fixed size vector (encoder), and then decoding a variable-length sequence from a fixed size vector. They claim the RNN encoder-decoder learns a continuous space representation of phrases that preserve both semantic and syntactic content.

The output of the encoder is the hidden state $c$ of the RNN at the last time step. For the decoder RNN, the hidden state at every time step is a function of the previous hidden state, the previous output, and $c$. That is, $h_{t} = f(h_{t-1}, y_{t-1}, c)$. The two components are then jointly trained to maximize the conditional normalized sum of the log likelihoods (the log likelihood of a single sequence-to-sequence translation is $\log{p_{\theta}(y_n | x_n)}$). The following image from the paper shows the broad overview of the architecture:

![RNN encoder-decoder](https://www.dropbox.com/s/pus9cwan90j6yy9/Screen%20Shot%202017-03-15%20at%202.02.10%20AM.png?dl=1)

The authors also introduce a new hidden unit similar to the LSTM. This consists of a reset gate $r_j$ and an update gate $z_j$. The reset gate decides how much of the previous hidden state to include in computing a temporary new hidden state $\tilde{h_t}$, which also depends on the input $x_t$, while the update gate decides how much information from the previous hidden state will carry over to the current hidden state. So: $$h_j^t = z_jh_j^{t-1} + (1-z_j)\tilde{h^t_j}$$

The RNN encoder decoder is applied in the scoring of phrase pairs in language translation. The statistical model of translation tries to find $f$ that maximizes $p(f|e) = p(e|f)p(e)$ (the translation and language model terms, respectively), given an input $e$. Phrase pairs from the two languages can be fed into the system, and the score is simply $p_{\theta}(y|x)$, where $(x, y)$ is the phrase pair. This score can then add as an additional feature in the model.

The authors also mention the use of [CSLM](https://ufal.mff.cuni.cz/pbml/93/art-schwenk.pdf) in their models, which uses NNs for the language model. It appears that the contributions of the RNN encoder-decoder and CSLM are independent. The authors claim that the embeddings generated by the encoder also capture both syntactic and semantic content of the phrases.



### **Sequence to Sequence Learning with Deep RNNs**

Paper: Sutskever et al. (http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

This paper is very similar to the paper on RNN encoder-decoders by [Cho et al](https://arxiv.org/pdf/1406.1078.pdf). The authors use deep RNNs with LSTMs as the hidden layer for the task of machine translation, essentially a sequence-to-sequence learning task.

The authors first use a deep (four layer) RNN with LSTMs for converting the input sequence into a fixed-dimensional vector (the hidden state of the final time step in this RNN), and then connected a second similar RNN to generate the output sequence (given the previous hidden state and previous emitted symbol). The "previous hidden state" for the first time step in the second RNN is the final hidden state in the first RNN.

An important strategy used by the authors is that they reverse the order of words in the input sequence. The intuitive reason why this improves results is as follows: consider an input sequence $a, b, c$ and the corresponding output $\alpha, \beta, \gamma$. If the input sequence is reversed so that the input-to-output mapping is now $c, b, a - \alpha, \beta, \gamma$, $a$ is in close proximity to $\alpha$, $b$ fairly close to $\beta$, and so on. Thus, it is easier for SGD to `` "establish communication" between the input and the output". The authors say that they do not have a complete explanation for this phenomenon, however.

Decoding the output sequence is done in the paper using beam search. The authors use a form of gradient clipping to address exploding gradients. They also made sure that all sentences within a minibatch were roughly of the same length, so that the more frequent shorter sentences do not suppress the learning of longer sentences within a minibatch.

The main way this paper differs from the paper by Cho et al. is that the RNNs are used directly for machine translation in this paper. In the other paper, the RNNs were used to obtain scores for phrase pairs, which made up the translation model of a Statistical Machine Translation system.



### **Attention-based Neural Machine Translation**

Paper: https://arxiv.org/pdf/1508.04025.pdf

This paper deals with techniques for applying "attention"-based mechanisms to RNNs in machine translation. The basic architecture used here is similar to that in the paper by [Sutskever et al. (2014)](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), with a stacking (deep) RNN with LSTMs used for encoding the source sentence, and then another connected deep RNN with LSTMs for producing the target sentence. The authors introduce two types of attention - global attention (all source words are attended) and local attention (a specific window of source words is attended).

In both types of attention approaches, at each time step $t$ of the decoding phase, a context vector $c_t$ is built. $c_t$ encodes the source-side "attentional" information. This is concatenated to the current hidden state of the decoder, $h_t$, and passed through a $tanh$ layer to produce a new hidden state, $\tilde{h_t}$ ($\tilde{h_t} = tanh(W_c[c_t;h_t])$).

* **Global attention:** In this method, all hidden states of the encoder are used to make $c_t$. First, a variable length alignment vector $a_t$ is made by assigning an alignment score $score(h_t, \bar{h_s})$ to the hidden state $\bar{h_s}$ of every source side word $s$, and then setting $a_t(s)$ as the softmax score. $c_t$ is then a weighted average of all the source hidden states, with the alignment vector as weights. Several different candidates are considered for the $score()$ function:

	* $h_t^T\bar{h_s}$ (dot)
	* $h_t^TW_a\bar{h_s}$   (general)
	* $v_a^Ttanh(W_a[h_t;\bar{h_s}])$ (concat)

The picture from the paper helps to clarify this:

![Global attention model](https://www.dropbox.com/s/xcjh625ojtn3it5/Screen%20Shot%202017-03-17%20at%2012.03.42%20PM.png?dl=1)

* **Local attention:** To reduce expense, local attention only considers a specific window of source words for every output time step. For every target word at time step $t$, the model generates an alignment position $p_t$, and only the set of source words in the window $[p_t-D, p_t+D]$ are used to calculate $c_t$ (similar to above). The authors use two methods to select $p_t$, the more successful being **local-p**: $p_t = S \cdot \sigma(v_p^T tanh(W_ph_t))$, where $W_p$ and $h_t$ are model parameters, and $S$ is the source sentence length. To attach more weight to words nearer to $p_t$, the authors attach a Gaussian weight to the scores, so $a_t(s) = align(h_t, \bar{h_s})exp(-\frac{(s-p_t)^2}{2\sigma^2})$, where the standard deviation is set to $\frac{D}{2}$, and $s$ is an index of a word within the window. The model is differentiable.

So far, the model does not take into consideration previous alignment information when generating target words. To address this, the authors use an input feeding approach, where the $\tilde{h}_t$ from the previous time step in the decoder is concatenated to the input of the next time step. This makes the model aware of previous alignment choices.

During training the authors pre-process the corpus by replacing words outside of the 50K most frequent with $<unknown>$ tags. The source sentence is reversed, gradient clipping used, and dropout ($p=0.2$) is employed between the spatial layers. $D$ is set to 10 for the local attention models,
