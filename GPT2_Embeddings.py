import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


st.set_page_config(page_title="Inside GPT-2 - Embeddings", layout="centered")  # Optional: sets browser tab title, layout

# st.title("Exploring Inner Representations in GPT-2")
st.markdown("<h1 style='text-align: center;'>Exploring Inner Representations in GPT-2</h3>", unsafe_allow_html=True)

st.markdown("""
## Introduction
<div style="text-align: justify;">

What follows below is an exploration of how words (tokens) are represented inside an AI language model like ChatGPT.

Before I go any further, I should say that this is not a tutorial or guide — just me sharing my own exploration and learnings.  
If anything is confusing, I apologize in advance. But all you really need to know is:  
> **I'm exploring how the internal representation of a word changes as it moves through a transformer based language model.**
Here, I have chosen to use the smallest version of GPT-2 (there are several larger versions) for convenience, which contains 12 transformer blocks.

### How do words move through the model?

In any transformer model like GPT-2, Words are first encoded as a sequence of tokens (numbers) through a process called *tokenization* — necessary because computers cannot process raw words directly.  
This step is handled by a **tokenizer**. Once tokenized, the numbers are passed through an **embedding layer**, which maps them into a high-dimensional vector space. How high-dimentional? 
The model that I am using here (GPT-2) projects each word into a 768 dimensional vector. But state of the art models use a much higher dimensional space (eg. ~12288 in GPT-3). So one can think of each word as being represented as a vector 
pointing in a certain direction with a certain magnitude (length) in this embedding space. 
These embeddings are then propagated forward through the model to predict the next word. 

A simple way to think about what happens inside the model might be:  
> As the embeddings move through each layer of the model (each transformer block), they are adjusted and updated so they become 'more similar' to the vector of the correct/likely next word, given the preceeding context.

This made me wonder whether it might be possible to **visualize** these internal changes as embeddings evolve through the transformer blocks. That's exactly what this visualization tries to do.

---

To accomplish this, I performed a Singular Value Decomposition (SVD) of the full embedding matrix and projected the data onto the first three principal components. So now the embeddings can be visualized as vectors in a 3-Dimensional space.
For visual clarity, I also **normalized** the vectors to unit norm, so they all have a length of 1, and I plot the norm of the vector separately in the lower panel. 
<div>
""", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>Example 1</h3>", unsafe_allow_html=True)

st.write("Now lets track the movement of the word 'first' through the following sentense:")

st.markdown("""
<p style='text-align: center; font-size: 20px; color: grey;'>
<b><i>"This is the very first example sentense."</i></b>
</p>
""", unsafe_allow_html=True)

st.image("embedding_with_similarity_and_norms0.gif")


st.subheader("Understanding the Embedding Trajectory Visualisation")

st.markdown("""
#### Top Panel: 3D Embedding Trajectory
<div style="text-align: justify;">

The top plot shows the **embedding vectors** reduced to 3 dimensions using the **SVD**. What this plot is really depicting is how the **direction** of the target embedding (in red) evolves as it moves through each of the 12 transformer blocks.

> **Note:** This plot only reflects the _directional changes_ in the (unit) normalized embeddings.

---

#### Lower Panel (Left): Embedding Vector Magnitude

This plot tracks the **norm (length)** of the target embedding across transformer blocks. It shows how the **magnitude** of the vector changes, as it moves through the transformer blocks.
It seems as though the magnitude of the vector consistently grows until the penultimate transformer block, and then smarply drops down at the output of the last transformer block. 

---

#### Lower Panel (Right): Similarity to Next Word

Here I am plotting two similarity metrics between the **target word embedding** (in red) and the **next word embedding** (in green):

- **Cosine Similarity (blue):**  
  Measures the _angle_ between the two vectors, regardless of the magnitude of the vectors. A value closer to 1 means they point in nearly the same direction.
  It seems like there is no appreciable increase in this similarity measure between the two vectors. There is a slight increase, but nothing too significant as far as I can tell. 

- **Dot Product (red):**  
  Measures the _magnitude of the projection_ of the target word embedding onto the next word embedding. This measure depends on both the **direction** and the **magnitude** of the target vector.
  We can see that the dot product, like the magnitude of the vector, grows larger until the penultimate transformer block, and then drops at the very last transformer block.
---

#### What does this tell us?

There's some useful information embedded in these plots. First, while the direction of the target embedding clearly changes across layers, it's important to realize that direction alone isn't the whole story. Admittedly, watching the 3D trajectory evolve is visually striking — but given that it represents a drastic compression from 768 dimensions down to just 3, we should be cautious about overinterpreting it. Still, the relatively modest growth in cosine similarity through the model is quite telling. This becomes especially insightful when you consider how language models actually predict the next word: by computing the dot product between the final target embedding and all other embeddings in the model's vocabulary. The words with the highest dot products are deemed most likely.

This means that both the direction and the magnitude (norm) of the final embedding vector influence which next word is selected. You can think of the dot product as the **shadow** the target embedding casts onto the embedding of the next word. So, even if two vectors are pointing in nearly the same direction (i.e., high cosine similarity), a shorter vector can still yield a smaller dot product — and thus a lower likelihood of that word being chosen.
</div>
""" , unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>Example 2</h3>", unsafe_allow_html=True)

st.write("Now lets take a look at how the word 'fleeting' travels through the embedding space in this sentense:")

st.markdown("""
<p style='text-align: center; font-size: 20px; color: grey;'>
<b><i>"Our brains turn fleeting sensations into the stories we live by."</i></b>
</p>
""", unsafe_allow_html=True)
st.image("embedding_with_similarity_and_norms2.gif")


st.markdown(
""" 
<div style="text-align: justify;">
The evolution of the vector norms, dot products and cosine similarities seem to be almost identical to that of the target word in the previous sentense.
If anything, the cosine similarity of the embedding vector with that of the next word, marginally decreases. This is a small sample size, but perhaps an indication that this is a repeating pattern?


### Starting position of the embedding
 --- Notice that the starting position of the embedding (red line) doesnt overlap with the actual embedding (blue line) that is taken directly from the embedings matrix. 
 This is because the red line is what the model sees before it enters the first transformer block, and there is an important step of adding something called the positional embeddings to the 
 actual word embedding before it enters the transformers. This addition already nudges the embedding off its starting position, but is an important step in telling the model about the position of the embedding in the full sequence.

### Are all tokens roughly pointed in the same direction? 
--- From visually inspecting the embeddings of each of the above sentenses, it seemed to me like the vectors all may be pointing roughly in the same direction. But this is a rediculously small sample 
based on which to make that inference. One way of quantifying this for the full embeddings matrix would be to compute a matrix of pairwise dot products between all the embeddings vectors. 

### Other questions this raises?
 This also raises quite a few other questions about the nature of the embeddings matrix. For instance, if all the vectors are more or less pointing in a similar direction, do they occupy a lower dimentional subspace 
 within the full embedding space? If so, what what is the rank of this embeddings matrix? Or at least what is its effective dimensionality? 


### Where to next?

This little exploration opened up a few more questions. The possibility that embeddings might live in a lower-dimensional subspace is intriguing — and raises fundamental questions about how these models compress, organize, and use semantic information.

An interesting next exploration might be analyzing the structure of the embedding matrix itself — its rank, the distribution of vector norms, clustering tendencies, etc. That might be better left for a follow-up post.

So, maybe next time:
What does the embedding space really look like?
And can we learn something about the language model just by studying it?

Stay tuned :)
</div>
""" , unsafe_allow_html=True)





st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <img src="Photo.png" width="100"/>
    </div>
    """,
    unsafe_allow_html=True
)st.markdown(
    "<p style='text-align: center;'>"
    "Created by <a href='https://github.com/Neojaltare' target='_blank'>Ketan Jaltare</a>"
    "</p>", unsafe_allow_html=True)
