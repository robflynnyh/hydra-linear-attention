# hydra-linear-attention
Implementation of the thingy described in this paper: https://arxiv.org/pdf/2209.07484.pdf

- code is mostly taken from the appendix of the paper its pretty simple 
- basically its linear attention with heads equeal to the feature dim, they use l2 norm as the kernel fn rather than softmax as it allows you to scale the "head" dimension, which makes it faster
- idk if it's descriptive to say stuff like this is similar to regular attention - I see it being more similar to something like squeeze and excite layers
