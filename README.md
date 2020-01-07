# Data-Modeling
Different basic approaches to modeling conditional stochastic datasets


### Deterministic
Trains a deterministic neural net to output a data matching that of the target distribution. Minimizes mean squared error between samples from the target distribution and samples from the network in order to train.


### Gaussian
Assumes that the data distribution at each point is Gaussian, allowing us to use a simple expectation maximization scheme. Is able to pick up basic nonlinearities well, but struggles if the distributions of data vary enough as you move along the x-axis (this shouldn't be a problem with Sense).


### Generative
Uses the Wasserstein-GAN (+ gradient penalty) algorithm to create a generative model of the data. This seems to fit the data best, but it takes much longer than the other methods (as is expected). If we ever get into environment modeling with Sense, we can always fall back on something like this if other models that make too many assumptions about the data fail.




# Usage

### Reading the code
There's some boilerplate code in each file, but the important stuff that actually does the training is wrapped in `########################################`

I also found that normalizing inputs to the network (i.e. using `input/max_input` instead of just `input`) resulted in *much* faster convergence when the data was nonlinear. If the inputs get too big, the model starts off linear and stays that way for a long time.


### Running the code
Make sure visdom is running. Then from the top level of the repo, run `python [filename].py`, where `filename` is one of the three scripts at the top level.

To change the data, go to the file `utils/data.py`. The function `req_to_instances` controls the target data distribution. I have two in there already: One is a square root function (# of instances = sqrt of # of requests +- constant variance) and a "fountain spray" function (concave down parabola with increasing variance as the # of requests increases).

:fire: Patrick/Andrew: it could be useful to try out the wacky function you were making in here to see what types of models work best with that structure, whether input normalization to the networks improves performance, how deep networks need to be to underfit/overfit, etc. :fire:


### What all the different utils files do
`data.py`: functions relating to the target data distribution. `plot_data()` plots samples from the target distribution. `req_to_instances()` is the function underlying the data distribution. `sample_states()` samples states uniformly. `sample_data()` samples states uniformly, plus a single sample of the output for each state

`env.py`: neural networks. `F` and `G` are the discriminator and generator, respectively, for the generative script. The others are pretty straightforward.

`vis.py`: functions for plotting. You should only ever need `scatter()` and `line()` (for scatter plots and line plots, respectively).


### Papers to read
Read the [Wasserstein GAN paper](https://arxiv.org/abs/1701.07875) and [WGAN with gradient penalty paper](https://arxiv.org/abs/1704.00028) if the generative modeling script doesn't make sense. The other scripts are pretty basic so I didn't reference papers for them.


### Results
There are some images in the `results` folder that show my visdom terminal after running each script on the "square root" data distribution. I got pretty similar results across trials, so you can use these as a baseline (my testing wasn't *that* extensive, though, so take my word with a grain of salt).