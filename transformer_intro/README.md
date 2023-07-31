# Transformer
The transformer is arguably the most influential model of the past decade.  It utilizes the attention mechanism so that the attention scales with the available computation, so that, in theory, a transformer's possible attention is unbounded.

** Continue intro **

## Architecture
If you've ever looked up a tutorial on transformers, you've likely seen this image from the seminal paper "Attention is All You Need" (CITE):

<img width="298" alt="Screen Shot 2023-06-20 at 3 23 50 PM" src="https://github.com/ArjunSohur/transformergallery/assets/105809809/1123363a-f956-450e-abc2-70909c555651">

The problem with this picture is that, unless you already know the ins and outs of a transformer model, the picture can be very confusing.

In the following article, we will traverse this scary picture and try to make it seem like common sense.  Note, however, that the encoder-decoder model is just a type of transformer.  The tag of "transformer" can extend to any machine learning model that contains some kind of encoder or decoder, but is not limited to including both.  Though transformers are far more flexible than their original encoder-decoder model; nevertheless, the encoder-decoder is the perfect place to start for learning purposes.

At a high level, the encoder's job is to produce a vector-representation of the input sequence, while the decoder deciphers the encoder's vector to produce an output.

Slightly more accurately, both the encoder and decoder are made up of many encoder and decoder blocks.  The encoder runs through its blocks to create a context vector for each word, which are then fed to all blocks of the decoder.  Importantly, the decoder is auto-regressive, which means it produces one word at a time.  Therefore, the decoder runs through all its blocks to produce a word with the help of the encoder's context vectors, then runs again to produce the next word (in an NLP context), which is based on the inputs context vectors and the answer that it has generated at that point.  In this manner, the decoder is simply a next word predicter powered by context vectors.

With enough training, the transformer seems to perform human-like tasks, but, keep in mind that it's easy to anthropomorphize complicated models - at their core, transformers, like any machine learning model, are just probabalistic.

The reason why transformers are so effective in their predictions is their self-attention mechanism, which leads to a slight disclaimer: to understand the transformer, you need to understand self-attention and multi-head attention.  Knowledge of them will be assumed and used as a base in the following explanations.  We have an article outlining these attention mechanisms to a sufficient degree of rigor if you would like to learn about it or if you need to burnish your memory.

With that being said, let's dive in!

## Positional Encoding
Before both the encoder and decoder, the input embedding and output embedding must go through a positional encoding step.

Transformers are attention-based models, so they are actually agnostic about each word's position, and, therefore, the potentially important positional relationships between words.

The most simple form of positional encoding is simply assigning a positive integer to each word in the sentence.  This method does designate each word to a unique position, but it starts to get shaky when dealing with sequences of varying sizes, which can make training difficult.  If we train this way, it will make the model flounder on sequences of unseen lengths, since it won't know the significance of a position it has never seen.

Ok, then, what if we normalize that positional encoding score so that for a sequence of $n$ words, we just assign the word $i$ to $\frac{i}{n}$?

While this idea takes the word's position relative to the whole sequence into account, it starts getting hairy when you assign .2 to the 1st word of a sequence of 5 and the 50th word of a 250 word long sequence.

The trick is to not avoid assigning each position a number, but instead use a fixed length vector that we can guarantee will be different in different positions.  The following is an algorithm that returns different vectors given a different position without creating the same vector twice:

To create a positional vector of dimension $d$ for position $pos$ in our sequence (which we will denote as $p_{pos} \in \mathbb{R}^d$), for $i$ s.t. $0 \leq i < d/2$:

$$ (p_{pos})^{(2i)} = \text{sin}(\frac{pos}{n^{2i/d}}) \text{ and } (p_{pos})^{(2i+1)} = \text{cos}(\frac{pos}{n^{2i/d}})$$

Where:
- $(p_{pos})^{(k)} \in \mathbb{R}$ denotes the scalar corresponding $k$th dimension of $p_{pos}$ (or the $k$ th element in the $p_{pos}$ array to use CS verbiage)
- $n$ is some predetermined number; the original transformer authors set it to $n=10000$

Let's think about how is happening for a bit.  We are creating a vector for the arbitrary position $pos$ of our sequence, which we are calling $p_{pos}$.  The values of $p_{pos}$ will be defined two at a time with index variable $i$ (we bound $i < d/2$ , where $d$ is the dimension or length of $p_{pos}$, so that we stay in the bounds of $p_{pos}$ since one $i$ will define two values of $p_{pos}$ at a time).

It is important to see how our formula produces entirely different vectors of dimension $d$ for every position.  The trick lies in the behavior of the sinusoidal functions and the clever way the formula was set up.  Without launching into a trigonometry lesson, putting $pos$ in the numerator of the angle of the sinusoidal function (along with the rest of the formula) guarantees that no two position's formulas are the same.  If you want to play with the variables of the formula, a well as get a better intrinsic understanding as to why the last statement is true, check out this graph:

https://www.desmos.com/calculator/x5qk7e5dut

In the end, we add these positional vectors to our preexisting word embeddings so that our transformer has information about the position of each word built into each vector.

Now, we should understand the motivation and technique behind this part of the graph:
<img width="514" alt="Screen Shot 2023-06-23 at 5 13 04 PM" src="https://github.com/ArjunSohur/transformergallery/assets/105809809/4c422c71-57cb-4f05-a8b7-55c9d44360ae">

** Maybe label the graph a little; how would people know that the arrows going into the plus represent? **

## Encoder
After positional encoding, we move on to understanding the encoder.  

To understand the encoder - and decoder, for that matter, - you must understand multihead attention.  If you are unfamiliar with the concept of multihead attention, or need to polish your memory, we have a comprehensive guide to attention in this gallery.  Furthermore, we provide resources that are helpful in our own learning process.

The good news is that once you understand multihead attention, the encoder starts to feel less daunting.

After doing positional encoding, we get an input that is of shape $\mathbb{R}^{n \times d}$, where $n$ is the input sequence length, and $d$ is the hidden encoding dimension.  A copy of this input matrix is stored while the input matrix itself goes through multi-head attention.  After we've completed multi-head attention, we add the copy of the input to the attentin output.  This steps battles against gradient vanishing or exploding and unpredictable, jagged gradient descent.

The last step of an encoder block is to clean up the input of the normalized and stabilized multi-head attention output.  We clean up by sending the multiple matrices through a fully-connected neural network that compares the outputs of each attention head and spits out one matrix of shape $\mathbb{R}^{n \times d}$, which can then functions as the input to the next layer of encoder.

Encoders, like classic neural networks, are usually stacked on one another for maximum learning.

In an encoder-decoder architecture, then, when we've iterated over all encoder blocks, we pass the results of the encoder, a context vector for each word, to each decoder block.  The context vectors aid the decoder in getting proper knowledge of the input to produce the output.


## Decoder

Whereas the encoder's job is to contextualize the input, a decoder's job is to produce the output.  The decoder simply predicts the next word as many times as it needs to produce an output.  Notably, the decoder is autoregressive, meaning it only is privy to whatever it has previously outputted.

The autoregressive nature of the decoder explains why we perform masked multi-head attention at the beginning of each decoder block; we don't want the decoder trying to create attention relationships with positions and tokens that have yet to be outputted.  To perform masked attention, we take a matrix the size of the input tensor with zeros in the lower triangle and $- \infty$ in the strict upper triangle.

Recall our formula for dot product attention:  $$S = f(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_{Q}}})V$$
Where: $Q \in \mathbb{R}^{n \times d_{Q}}, K \in \mathbb{R}^{n \times d_{K}}, V \in \mathbb{R}^{n \times d_{V}}$

Let's consider a mask as described above: $ M \in \mathbb{R}^{n \times n \text{ s.t. } M \text{'s}$ lower triangle (including the diagonal) is 0, but $M$ 's strict upper triangle is $- \infty$. Then our masked multi-head attention turns out to be:
$$S = f(Q,K,V) = \text{softmax}(\frac{QK^T + M}{\sqrt{d_{Q}}})V$$
Then, when run through the softmax, the attention matrix will be such that no token will have any context about its proceeding tokens.  Keep in mind that the decoder's task - next word prediction - means that it will never have any words that it hasn't already produced itself, which is the reason why we mask the attention at the start of the decoder.

Much like after the multi-head attention in the encoder, we add the input and normalize the resulting matrix to help gradient descent.

Next in the decoder, we pass another attention block, but this time, there is no mask. Instead, we use the context vector that the encoder produced to help predict the next output word.  In the transformer's description, we see an arrow from the top of the encoder going to this attention block in the decoder; the arrow represents the aforementioned context vector.  From the encoder's context vector, the decoder derives its keys and values for the attention layer.  Its queries come from the masked attention block.  From there we computer multi-head attention normally, including the add & norm part.

As with the encoder layer, we send the final result of multi-head attention through a feed forward neural network to ready our attention output to be the input to the next layer.

## Conclusion
Perhaps you need to hear the transformer explained from many sources, but we hope that we've provided a clear idea of the architecture of the transformer.  As a quick refrence, we've provided a marked-up version of the picture at the start of the article:

<img width="394" alt="Screen Shot 2023-07-30 at 10 07 49 PM" src="https://github.com/AIResearchHub/transformergallery/assets/105809809/d7b8d746-e44d-4cf1-baba-ef0592d5abd7">


### Sources
https://machinelearningmastery.com/the-transformer-model/
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/#:~:text=What%20Is%20Positional%20Encoding%3F,item's%20position%20in%20transformer%20models.
https://medium.datadriveninvestor.com/transformer-break-down-positional-encoding-c8d1bbbf79a8
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
https://www.turing.com/kb/brief-introduction-to-transformers-and-their-power#the-transformer-encoder
https://data-science-blog.com/blog/2021/04/07/multi-head-attention-mechanism/





