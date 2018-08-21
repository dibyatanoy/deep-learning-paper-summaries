<br>
<br>
<br>

This is a list of summaries of papers in the field of deep learning I have been reading recently. The summaries are meant to help me internalize and keep a record of the techniques I have read about, and not as full analyses of the papers.

# Table of contents
* [Emergence of Invariance and Disentanglement in Deep Representations - Parts 1 and 2](#emergence-of-invariance-and-disentanglement-in-deep-representations-part-1)
	* Alessandro Achille, Stefano Soatto (2018)
* [Speech Recognition With Deep Recurrent Neural Networks](#speech-recognition-with-rnns)
	* Graves, Mohamed and Hinton
* [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](#rnn-encoder-decoders-in-statistical-machine-translation)
	* Cho, van Merrienboer, Gulcehre, Bahdanau, Bougares, Schwenk and Bengio
* [Sequence to Sequence Learning with Neural Networks](#sequence-to-sequence-learning-with-deep-rnns)
	* Sutskever, Vinyals and V. Le
* [Attention-based Neural Machine Translation](#attention-based-neural-machine-translation)
	* Luong, Pham and Manning (2015)
* [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](#dynamic-memory-networks-for-nlp)
	* Kumar et al. (2016)
* [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](#spatial-pyramid-pooling-in-deep-cnns)
	* Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
* [Neural Turing Machines](#neural-turing-machines)
	* Alex Graves, Greg Wayne, Ivo Danihelka (2014)

<br>

### **Emergence of Invariance and Disentanglement in Deep Representations (Part 1)**

Paper: https://arxiv.org/pdf/1706.01350.pdf

This paper looks at the broad question of "Why do heavily over-parametrized deep nets generalize well?" from the perspective of Information Theory. This was the first paper I've read fully that takes an information theory approach, and was quite a long read.

First, it's useful to know some terms related to representations. $z$ is a _representation_ of $x$ if "the distribution of $z$ if fully described by the conditional $p(z|x)$", giving rise to the Markov chain $y \rightarrow x \rightarrow z$, $y$ being the task. $z$ is **sufficient** for $y$ if $I(z;y) = I(x;y)$ (remember that mutual information, informally, is a measure of how much information $x$ captures about $y$) . $z$ is **minimal** when $I(x;z)$ is smallest among sufficient representations (we don't want to memorize too much). A **nuisance** $n$ is something that affects $x$ but is "not informative to the task we're trying to solve", i.e., $I(y;n)  = 0$. A representation $z$ that minimizes $I(z;n)$ among all sufficient representations is said to be **maximally insensitive** to $n$. The [Total Correlation (TC)](https://en.wikipedia.org/wiki/Total_correlation) of a distribution is defined as $TC(z) = KL(p(z) || \Pi_{i} p(z_i))$ - note that TC can also be regarded as the amount of (redundant) shared information among variables in a set. A useful relation to remember is $I(x;y) = KL(p(x, y) || p(x)p(y))$ - the expected extra (redundant) number of bits to identify $x$ and $y$ if they are transmitted using their marginal distributions instead of their joint distributions.

One of the first interesting facts in the paper is that invariants (think nuisance-free representations) can be constructed by reducing the mutual information between $x$ and $z$, i.e., minimality. The authors state that if $n\rightarrow x \rightarrow z$, then $I(z;n) \leq I(z;x) - I(x;y)$. The second term is a constant, so the lower the value $I(z;x)$, the more invariant the representation $z$. Bottlenecks, such as dimensionality reduction between successive layers of a network, also promote invariance - if $x \rightarrow z_1 \rightarrow z_2$, and there is a communication bottleneck between $z_1$ and $z_2$, then provided that $z_2$ is still sufficient, $z_2$ is more nuisance-invariant than $z_1.$ This also implies that stacking layers ("deep nets") promotes invariance, although this does not simply mean that more layers means better generalization, since it assumes that the last layer is still a sufficient representation for $x$ (meaning the network has been trained properly, which is increasingly difficult for large networks).

The next part of the paper states that the amount of information in the weights can act as a useful regularizer, allowing us to control when a network will overfit/ underfit. The paper decomposes the standard cross-entropy loss into the following form: $$H_{p, q}(x, y) = \mathrm{(sum\ of\ positive\ terms)} - I(y;w| \mathbf{x}, \theta)$$ The only negative quantity here is the last term, which can be thought of as how much information the weights have "memorized" about the labels (since we are already conditioning on the true state of nature and the dataset). Thus, a network _could_ minimize this loss simply by increasing the term on the right, leading to overfitting, i.e., by memorizing the dataset. We would thus want to add a term back into the loss to account for this, but $I(y;w|\mathbf{x}, \theta)$ is intractable. We can still upper bound this term, by noticing that $I(y;w|\mathbf{x}, \theta) \leq I(w;\mathcal{D}|\theta) \leq I(w; \mathcal{D})$ (since $\theta \rightarrow \mathcal{D}$, and conditioning reduces mutual information for a Markov chain). Thus, we can write the new loss as $L = H_{p, q}(\mathbf{x}, y) + \beta I(w;\mathcal{D})$. Apparently, this was suggested as a regularizer as far back as 1993 by Hinton, but no efficient way to optimize this was known until [Kingma's paper](https://arxiv.org/pdf/1506.02557.pdf) in 2015. The authors also further upper bound this term, showing that $I(w; \mathcal{D}) \leq KL(q(w|\mathcal{D}) || \Pi_{i} q(w_i))$, where $q(w)$ is a distribution over the weights over all possible trainings and datasets. This is then used in the local reparametrization trick.

Interestingly, the $\beta$ term can also be used to predict precisely when overfitting or underfitting will occur for random labels, as shown below (figures from the paper). By changing the value of $\beta$, which controls the amount of information in the weights, the authors also obtain a graph that closely resembles the classing bias-variance tradeoff curve, suggesting that $\beta$ (and thus, the information in the weights) correlates well with generalization.

<center>![Train and test accuracies for different values of beta](https://www.dropbox.com/s/r1ua7avmxugwzbi/Screen%20Shot%202018-08-20%20at%203.29.49%20PM.png?dl=1)</center>

<center>![Test error vs beta - resembles bias-variance tradeoff curve](https://www.dropbox.com/s/nurqe525o8zfduf/Screen%20Shot%202018-08-20%20at%203.30.24%20PM.png?dl=1)</center>

The paper also mentions that under certain conditions, Stochastic Gradient Descent, without this regularizer, "introduces an entropic bias of a very similar form to the information in the weights" that was described above. Additionally, the authors also note that some forms of SGD bias the optimization towards "flat minima", which require lower $I(w;\mathcal{D})$. This could explain why even without this regularizer, networks can often be trained to be generalizable. Note that it is commonly believed that SGD implicitly acts as a regularizer, due to the noise introduced by the stochasticity.

The second part of the writeup will deal with the remaining results in the paper.




### **Emergence of Invariance and Disentanglement in Deep Representations (Part 2)**

Paper: https://arxiv.org/pdf/1706.01350.pdf

Sections 4 and 5 of this paper derive more interesting results about the relationship between information in the weights, flat minima, and sufficiency. Using modeling assumptions on the weights, the authors upper bound the information in the weights similarly to [Kingma et al. (2015)](https://arxiv.org/pdf/1506.02557.pdf). They use an improper log-uniform prior on $w$ ($\tilde{q}(w_i) = c/|w_i|$), and parametrize the weight distribution during training as $w_i | \mathcal{D} \sim \epsilon_i\hat{w}_i$, where $\hat{w}_i$ is a learned mean and $\epsilon_i \sim \log \mathcal{N}(-\alpha_i/2, \alpha_i)$ is "i.i.d. multiplicative log-normal noise with mean 1 and variance $\exp(\alpha_i)-1$". Using this parametrization, they show an upper bound on the information in the weights $I(w; \mathcal{D})$, which is used with the local reparametrization trick as a regularizer.

The authors also formally show a relationship between $I(w; \mathcal{D})$ and the "flatness of the minima". Flatness is a vague term, and is formalized as "the nuclear norm of the Hessian of the loss function". While I only have a "visual intuition" of why this would work (smaller nuclear norm $\rightarrow$ smaller singular values of Hessian $\rightarrow$ less rapidly changing gradients?), I have yet to find a more rigorous justification of this. The authors show that $$I(w;\mathcal{D}) \leq \frac{1}{2}K[(\log ||\hat{w}||_2^2) + log||\mathcal{H}||_* - \mathrm{(some\  constant)}]$$ where $\hat{w}$ is the minima of the weights according to the cross-entropy loss, $\mathcal{H}$ is the Hessian, and $K =\  \mathrel{dim}(w)$. Thus, the flatter the minima, the lower the information in the weights!

In the next section, the authors prove that $I(x;z) + TC(z)$ is tightly bounded by $\tilde{I}(w;\mathcal{D})$ (this is itself a sharp upper bound for $I(w;\mathcal{D})$). This means that lowering the information in the weights (explicitly, or implicitly via SGD) automatically improves the minimality and hence, by earlier propositions (see part 1 above), the invariance and disentanglement of the learned representation. Using the Markov property of the layers, the authors extend this to multi-layer networks. Taken together with the facts stated earlier, this implies that SGD is "biased toward learning invariant and disentangled representations of the data".




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

<center>![RNN encoder-decoder](https://www.dropbox.com/s/pus9cwan90j6yy9/Screen%20Shot%202017-03-15%20at%202.02.10%20AM.png?dl=1)</center>

The authors also introduce a new hidden unit similar to the LSTM - the **Gated Recurrent Unit (GRU)**. This consists of a reset gate $r_j$ and an update gate $z_j$. The reset gate decides how much of the previous hidden state to include in computing a temporary new hidden state $\tilde{h_t}$, which also depends on the input $x_t$, while the update gate decides how much information from the previous hidden state will carry over to the current hidden state. So: $$h_j^t = z_jh_j^{t-1} + (1-z_j)\tilde{h^t_j}$$

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

<center>![Global attention model](https://www.dropbox.com/s/xcjh625ojtn3it5/Screen%20Shot%202017-03-17%20at%2012.03.42%20PM.png?dl=1)</center>

* **Local attention:** To reduce expense, local attention only considers a specific window of source words for every output time step. For every target word at time step $t$, the model generates an alignment position $p_t$, and only the set of source words in the window $[p_t-D, p_t+D]$ are used to calculate $c_t$ (similar to above). The authors use two methods to select $p_t$, the more successful being **local-p**: $p_t = S \cdot \sigma(v_p^T tanh(W_ph_t))$, where $W_p$ and $h_t$ are model parameters, and $S$ is the source sentence length. To attach more weight to words nearer to $p_t$, the authors attach a Gaussian weight to the scores, so $a_t(s) = align(h_t, \bar{h_s})exp(-\frac{(s-p_t)^2}{2\sigma^2})$, where the standard deviation is set to $\frac{D}{2}$, and $s$ is an index of a word within the window. The model is differentiable.

So far, the model does not take into consideration previous alignment information when generating target words. To address this, the authors use an input feeding approach, where the $\tilde{h}_t$ from the previous time step in the decoder is concatenated to the input of the next time step. This makes the model aware of previous alignment choices.

During training the authors pre-process the corpus by replacing words outside of the 50K most frequent with $<unknown>$ tags. The source sentence is reversed, gradient clipping used, and dropout ($p=0.2$) is employed between the spatial layers. $D$ is set to 10 for the local attention models.



### **Dynamic Memory Networks for NLP**

Paper: https://arxiv.org/pdf/1506.07285v1.pdf

This paper discusses a framework for question answering given some previous inputs. The dataset consists of triples: a list of sentences (inputs), a question using facts found in the sentence, and an answer. Facebook's bAbI dataset is an example. The framework is divided into a series of modules: input module, semantic memory module, question module, episodic memory module, and an answer module.

* **Semantic module**: The semantic module consists of word concepts and facts about them. In the paper, the module consists of embeddings for words in the form of Glove vectors, although the authors say the module could serve to store other knowledge as well.

* **Input module**: The input module uses an RNN with GRUs to convert the input word embeddings into a sequence of facts. Specifically, for every word $w_t^I$ in the input, the associated fact vector is $c_t = GRU(L[w_t^I], c_{t-1})$, where $L[w]$ denotes the word embedding for word $w$.

* **Question module**: The question module uses the same RNN (i.e., shares the embedding and GRU weights) to convert the question into a single question vector $q$, which is simply the final hidden state of the RNN when fed with the input ($q = q_{T_Q}$, where $T_Q$ is the length of the question).

* **Episodic memory module**: This is the most crucial part of the entire network. The goal is to produce a final memory vector $m$. $m$ is generated from a sequence of episode vectors $e_i$, each one made by making a pass over all the facts $c_t$ and using a soft attention mechanism to selectively attend to them. The module can be broken down into an inner GRU and an outer GRU:
	* Outer GRU: The outer GRU works on a sequence of episodes $e^i$, producing corresponding memory vectors. The GRU state is initialized with the question vector. The recurrence thus looks like: $m^i = GRU(e^i, m^{i-1})$. The final memory vector is the overall memory vector $m$.
	* Inner GRU: The inner GRU computes episodes. Each time an episode $e^i$ is generated, we need the attention mechanism to "assign weights" to each of the input facts $c_t$. The attention mechanism works like a gate, and for each fact $c_t$, it computes $g_t^i = G(c_t, m^{i-1}, q)$. $G$ is a sigmoid function over a series of matrix multiplications and tanh layers (see the paper for more details). Note that the gates at each pass depend on the memory vector $m^{i-1}$ from the previous episode. Once the gates are calculated, the inner GRU then uses these gates to compute a sequence of hidden states for episode $i$: $h_t^i = g_t^iGRU(c_t, h_{t-1}^i) + (1-g_t^i)h^i_{t-1}$. The episode vector is the final hidden state: $e^i = h^i_{T_C}$, where $T_C$ is the number of candidate facts.

* **Answer module**: The memory vector $m$ is fed into the answer module, which also consists of an RNN with GRUs. The initial hidden state $a_0 = m$.  The recurrence for the hidden state is then $a_t = GRU([y_{t-1};q], a_{t-1})$ and the output is $y_t = softmax(W^{(a)}a_t)$. The output can also be a special stop token denoting the end of the sentence. The model is trained with cross entropy error classification of the correct sequence appended with a special end-of-sequence token.

The model is trained with backpropagation and Adagrad. The authors use $L_2$ regularization and 'word dropout', where each word vector is set to $0$ with some probability $p$. The model can then perform tasks like question answering, POS tagging, sentiment analysis and even machine translation, better than a lot of existing models.



### **Spatial Pyramid Pooling in Deep CNNs**

Paper: https://arxiv.org/pdf/1406.4729.pdf

In this papers, the authors address the issue of allowing images in the input to have variable sizes (dimensions). Typically in CNNs, all images in the input are pre-processed (e.g.: by cropping/ warping) so that the images are all of the same size. This is needed not because of the convolutional layers (which operate in a sliding window fashion), but for the fully-connected layers. The authors add a spatial pyramid pooling layer after the final convolutional layer to allow inputs of arbitrary size.

As mentioned above, convolutional layers do not need fixed sized inputs. This is because the "filter" can be slid across the image/ feature maps with the appropriate stride until the entire image is covered. This also applies to the maxpooling layers that might be placed after the convolutional layers. In the paper, the spatial pyramid pooling layer is placed after the final convolutional layer. Suppose that the final convolutional layer has dimensions $d \times d \times k$, where $k$ is the number of filters (two of the dimensions are the same to make this example easier). Spatial pyramid pooling is analogous to creating a set of bins of varying size. The pyramid consists of 'level's, where each 'level' is like a grid laid out over each of the $d \times d$ filters. Each bin is like a square in these grids. For instance, if we want to create a $4 \times 4$ level, this level will give us $16$ bins, and the size of each bin will be $\lceil d/4 \rceil \times \lceil d/4 \rceil$ (with some bins possibly having parts outside the images). Within each bin, we can use a pooling operation (the paper uses maxpooling). This is similar to creating a maxpooling layer with with the dimension of each pool as $\lceil d/4 \rceil \times \lceil d/4 \rceil$, and a stride of $\lfloor d/4 \rfloor$. Each of these levels is applied to each layer in the filter. If we create $M$ bins in total across all the levels, the output of this layer will thus be $M$ $k-$dimensional vectors. This is illustrated in the following figure from the paper:

<center>![SPP layer](https://www.dropbox.com/s/56g6pee33vu7276/Screen%20Shot%202017-03-28%20at%201.25.32%20AM.png?dl=1)</center>

In this figure, there are 3 levels, $M = (16 + 4 + 1) = 21$, and $k = 256$.

The authors state that current GPU implementations are preferably run on fixed input sizes, so at training time, they consider a set of predefined sizes. For each of these predefined sizes, a different network is used (all the networks share the same parameters, however, and the number of parameters is the same because the output of the  SPP layer is the same size for all networks).  In  other  words,  during training they implement the varying-input-size SPP-net by two fixed-size networks that share parameters.



### **Neural Turing Machines**

Paper: https://arxiv.org/pdf/1410.5401.pdf

The authors in this paper augment neural networks by adding the analog of an "addressable memory" to the system. The network consists of a controller (think CPU) that interacts with the external inputs and produces outputs, and a "memory matrix" (think RAM) that the controller can read to or write from via read/ write "heads". Since neural networks like RNNs using LSTMs only have a limited amount of memory, the idea is that having a dedicated memory matrix will make recalling information easier. The basic structure is as shown in the diagram below:

<center>![NTM layout](https://www.dropbox.com/s/kxbjbgjte6i2wpg/Screen%20Shot%202017-05-23%20at%2012.14.44%20PM.png?dl=1)</center>

There are 2 main parts that are new to this network: the reading mechanism and the writing mechanism.

* **Reading Mechanism**: We use $M_t$ to denote the contents of the memory matrix at time $t$. The memory matrix is an $N\times M$ matrix, consisting of $N$ memory locations each of size $M$. Reading is similar to applying an attention mechanism over all the memory locations/ vectors. Every read head outputs a vector $w_t$ of normalized read weights, such that $\sum\limits_{i}w_t(i) = 1$. The length $M$ read vector $r_t$ produced by this head at time $t$ is then a weighted average of all the memory locations:

   <center>$r_t = \sum\limits_{i}w_t(i)M_t(i)$</center>

	This is the result of the read operation by this read head at time $t$.

* **Writing Mechanism**: The writing mechanism consists of 2 steps: an 'erase' followed by an 'add'. Every write head produces 1 length $N$ and 2 length $M$ vectors: a weighting vector $w_t$, an erase vector $e_t$ and an add vector $a_t$, respectively. The erase operation modifies the memory vectors from the previous step according to $\tilde{M_t}(i) = M_{t-1}(i)[1-w_t(i)e_t]$. The add operation then adds to these modified vectors, according to $M_t(i) = \tilde{M_t}(i) + w_t(i)a_t$.

* **Producing the Weights (Addressing Mechanism)**: The weights produced by both the read and write vectors are via a combination of content-based addressing (pick out elements based on content) and location-based addressing.
For the content-based addressing, each head produces a length $M$ key vector $k_t$: the "content" that we want to search for. Each memory location is then assigned a weight based on how similar it is to $k_t$, and the result of the content-based addressing is then a softmax-like weight calculation. Specifically, denoting the similarity mechanism (e.g.: cosine distance) as $K[\cdot ; \cdot]$, we can write:

	<center>$w^c_t(i) = \frac{exp(\beta_{t}K[k_t, M_t(i)])}{\sum_{j}exp(\beta_{t}K[k_t, M_t(j)])}$</center>

	where $\beta_t$, produced also by the head, is called the "positive key strength" and determines how sharply to reward similarity to the key vector.

  The location based addressing operates on top of these weights. Each head emits a scalar interpolation gate $g_t$, between 0 and 1, which blends between the weighting from the previous time step, and the current content-based weights.

 <center>$w_t^g = g_tw^c_t + (1-g_t)w_{t-1}$. </center>

 If this gate is $0$, the content-weighting is entirely ignored.
 After this step, each head emits a shift weighting $s_t$ which can be used for head rotation. The rotate operation is useful for sequential memory access, for example. If the range of valid shifts is $+/-k$, then $s_t$ is a length $N$ vector with $2k+1$ non-zero elements, corresponding to the degrees to which the shifts in the range $-k$ to $+k$ are allowed. The rotated weight vector is thus obtained by:
	<center>$\tilde{w_t}(i) = \sum\limits_{j=0}^{N-1}w^g_t(j)s_t((j-i) mod N)$</center>

 Every element $\tilde{w_t}(i)$ in the rotated weight vector is a blend of all the positions from which a valid shift to position $i$ exists, weighed by the appropriate shift vector weights. Finally to sharpen these weights, each head emits one scalar $\gamma_t$, which is used as follows:
 <center>$w_t(i) = \frac{\tilde{w_t}(i)^{\gamma_t}}{\sum_j \tilde{w_t}(j)^{\gamma_t}}$</center>


The controller can use either a feed-forward neural net or a recurrent neural net. If an RNN with an LSTM is used, the memory of the LSTMs can be likened to the registers of a CPU, allowing data from multiple times steps to be stored and used together.

The NTM is evaluated on algorithmic tasks such as copying, repeat-copying, associative recall, dynamic N-grams and priority sort.

**N.B.**: What are Hopfield networks?
