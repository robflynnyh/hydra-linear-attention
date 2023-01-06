# hydra-linear-attention
Implementation of the thingy described in this paper: https://arxiv.org/pdf/2209.07484.pdf

- code is mostly taken from the appendix of the paper its pretty simple 
- basically its linear attention with heads equeal to the feature dim, they use l2 norm as the kernel fn rather than softmax as it allows you to scale the "head" dimension, which makes it faster


- idk if it's descriptive to say stuff like this is similar to regular attention - I see it being more similar to something like squeeze and excite layers
- This uses the values from each individual dimension to create a vector that is the size/height of the sequence length for each dimension
- And then uses a similarity fn to get a weighting for each dimension in the input
- so basically u can use the global context of the input to downregulate useless dimensions upregulate others - but every element in the sequence receives the same transformations
- I imagine that makes it quite useful for letting the network remove irrelevent stuff/noise from the input in domains like speech/vision

