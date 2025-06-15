# Graph Neural Networks from Scratch with JAX

![JAX and Equinox illustration](./.illustration.png)

This repo is an implementation example of GNNs using JAX. The GNN is the simplest graph
convolutional network, implemented either with the adjacency-based or the edge-based representation.

Those two implementations should give the reader the keys to implement its own GNN. You can also
have a look at the following [blogpost][blogpost] for a more in-depth presentation.

## Training objective

The code trains the GNN to rank the nodes of randomly generated graphs using the
[nx.betweenness][betweenness-centrality] score. The [margin-ranking loss][margin-ranking-loss] is
used. The [Kendall tau][kendall-tau] is used to judge the correlation between the produced rank from
the model and the actual rank from the scores.

## Usage

This repo needs python 3.13 and `uv`. You can specify the hyperparameters by editing the
`./configs/default.yaml` file or by passing modifications in the commandline.

Simply launch the training with:

```sh
python3 main.py
```

The training is logged to WandB. Have a look to the [Hydra][hydra] (for HPs config) and
[WandB][wandb] (for logging) docs for more.


[betweenness-centrality]:   https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
[blogpost]:                 https://pierrot-lc.dev/posts/2024-09-02_jax-gnn/
[hydra]:                    https://hydra.cc/
[kendall-tau]:              https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
[margin-ranking-loss]:      https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
[wandb]:                    https://wandb.ai/
