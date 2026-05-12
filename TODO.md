# TODO

## Energy degeneracies

In practice of NS simulations at 32bit float precision, we often face the problem of degenerate walkers. If the highest energy walker happens to be degenerate with another walker, it needs a statistically sound way of deciding which one to cull. I think uniform sampling is required here, but at some point we need to carefully think about this.

