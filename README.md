# hydra-linear-attention
Implementation of the thingy described in this paper: https://arxiv.org/pdf/2209.07484.pdf
- Coudn't see any code anywhere for this, though its just linear attention with heads equeal to the feature dim, it's pretty short and simple, create an issue is incorrect as I only skimmed the paper,
idk if it's descriptive to say stuff like this is similar to regular attention - I see it being more similar to something like squeeze and excite layers


- This uses the values from each individual dimension to create a vector that is the size/height of the sequence length for each dimension
- Then computes cosine similarity with itself for each of those vecors - meaning ur left with a scaler for every dimension (i.e (B,D,1,1))
- This cosine similarity score is then used to weight each dimension by the scaler

- so basically u can use the global context of the input to downregulate useless dimensions upregulate others - but every element in the sequence receives the same transformations
- I imagine that makes it quite useful for letting the network remove irrelevent stuff/noise from the input in domains like speech/vision

