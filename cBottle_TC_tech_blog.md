# cBottle in Earth2Studio: Estimating Extreme Weather Likelihoods with Guided Generative Climate Models

Extreme weather risk often depends on events that are rare, localized, and expensive to sample with traditional climate simulations. Tropical cyclones, heat waves, atmospheric rivers, and other high-impact events may require very large ensembles before enough examples appear to estimate their likelihood with useful confidence.

A new research paper, [Towards accurate extreme event likelihoods from diffusion model climate emulators](https://arxiv.org/abs/2605.03802), shows that diffusion-based climate emulators can do more than generate realistic atmospheric states. They can also provide useful probability estimates for those states.

Using the cBottle generative climate model, the paper demonstrates how to guide a diffusion model toward tropical cyclone states and compute an odds ratio between the guided and unguided distributions. This enables importance sampling of rare events, reducing the uncertainty of probability estimates compared with simple Monte Carlo sampling.

An implementation of this workflow is now available in Earth2Studio through the [CBottleTCGuidance](https://github.com/NVIDIA/earth2studio/blob/main/examples/07_misc/05_cbottle_tc_guidance.py) example.

## Why estimating extreme event probabilities is hard

Climate and weather models can generate physically realistic atmospheric states, but rare events remain difficult to study statistically. If a tropical cyclone over a specific location is rare under a given climate state, unguided sampling may require many model draws before enough events appear.

This creates a bottleneck for estimating the likelihood of events such as tropical cyclone landfall, comparing event probabilities under different climate boundary conditions, studying tail risk for infrastructure and insurance, and generating plausible rare scenarios for downstream impact models.

Diffusion models provide a new path. Because they learn a probability distribution over atmospheric states, they can be guided toward rare events while still producing samples from a known modified distribution.

The key is not only to generate more extremes, but to know how much the guidance changed the probability of each sample.

## From guided samples to odds ratios

The cBottle tropical cyclone guidance workflow generates atmospheric samples conditioned on user-specified tropical cyclone activity. In the Earth2Studio example, the guidance tensor marks a desired location near Florida and requests a sample during hurricane season.

Conceptually, the model compares two probabilities for a generated atmospheric state $$x$$:

$$
p_\text{guided}(x)
$$

and

$$
p_\text{unguided}(x)
$$

The log-odds ratio is:

$$
\log r(x) = \log p_\text{guided}(x) - \log p_\text{unguided}(x)
$$

This value quantifies how much more likely the sample is under the guided model than under the base model.

For estimating probabilities under the unguided distribution using guided samples, the corresponding importance weight is:

$$
w(x) = \frac{p_\text{unguided}(x)}{p_\text{guided}(x)} = \exp(-\log r(x))
$$

This is what enables importance sampling: generate more samples in the rare-event region, then reweight them to estimate their likelihood under the original climate distribution.

## Running guided tropical cyclone sampling in Earth2Studio

The Earth2Studio example uses the built-in `CBottleTCGuidance` diagnostic model. The setup loads the default cBottle TC guidance package, which downloads the checkpoint from NGC.

```python
from earth2studio.models.dx import CBottleTCGuidance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the default model package which downloads the checkpoint from NGC
package = CBottleTCGuidance.load_default_package()
```

The fast path is optimized for standard guided inference. Here, a single guidance point is placed near Florida, using latitude $$27.0^\circ$$ and longitude $$-82.0^\circ$$. The example then requests one timestamp during hurricane season.

```python
lat = torch.tensor([27.0], device=device)  # Near Florida
lon = torch.tensor([-82.0], device=device)  # Converted internally to [0, 360)
times = [datetime(2005, 10, 11, 12)]

model = CBottleTCGuidance.load_model(package, seed=0).to(device)
# Create guidance tensor
guidance, coords = model.create_guidance_tensor(lat, lon, times)
guidance = guidance.to(device)
# Run guided sampling
guided_sample, guided_coords = model(guidance, coords)
```

The output, `guided_sample`, contains the generated atmospheric state. 

This makes it straightforward to extract variables such as 10-meter zonal wind, mean sea-level pressure, or other generated fields for analysis and visualization. The example plots the 10-meter zonal wind component, `u10m`, over a Caribbean domain (Fig. 1). This visualization helps verify that the generated state contains coherent tropical cyclone-like wind structure near the requested location.

![Fig1](/home/pmanshausen/earth2studio/guided_sample.png)

## Computing the log-odds ratio

Sampling a guided tropical cyclone is useful, but the main research contribution is the ability to estimate how likely that sample is under the guided distribution compared with the unguided base distribution.

The odds-ratio calculation requires calculating the divergence of the generative flow, which in turn needs second-order derivatives through the model. For that reason, the example reloads the model with `allow_second_order_derivatives=True`.

```python
model = CBottleTCGuidance.load_model(
    package,
    seed=0,
    sampler_steps=2,
    allow_second_order_derivatives=True,
).to(device)

log_odds_ratio, forward_latents, latent_coords = model.calculate_odds_ratio(
    guidance,
    coords,
)

print(f"Log odds ratio: {log_odds_ratio:.4f}")
print(f"Forward latents shape: {tuple(forward_latents.shape)}")
```

The example reduces `sampler_steps` to speed up runtime. For higher-quality samples and more stable odds-ratio values, the default sampler settings should be used.

The returned `log_odds_ratio` tells you how much the guidance changed the likelihood of the generated sample. A negative value means the sample is much more likely under the guided distribution than under the unguided base model.

The `forward_latents` tensor provides the guided sample used by the odds-ratio computation. 

## Why this matters for climate science

The paper shows that diffusion model climate emulators can provide probabilistic information that has been previously impossible to obtain for full atmospheric states.

This is important because many climate questions are fundamentally probabilistic. Researchers want to know how likely a tropical cyclone is near a specific coastline, how that likelihood changes under different sea surface temperatures, and whether guided generative models can produce rare events more efficiently than brute-force sampling.

By combining guided generation with odds-ratio diagnostics, we can sample rare events more often while still estimating their likelihood under the base climate distribution. In the paper, this enables importance sampling of tropical cyclone states and reduces standard error compared with simple Monte Carlo sampling.

## Opportunities beyond climate

Although the example focuses on tropical cyclones, the idea is broader: guide a generative model toward a rare but important part of the distribution, then compute how much the guidance changed the sample probability.

This pattern could be useful in domains where rare events dominate risk, including power grid resilience, aerospace stress testing, financial tail-risk estimation, materials discovery, robotics edge cases, and molecular simulation.

When naive sampling rarely finds the important event, guided generative sampling plus odds-ratio correction can make rare-event estimation more practical.

## Future research directions

The current results are early but promising. More work is needed to make these methods faster, more accurate, and more broadly applicable.

Diffusion sampling can be computationally expensive, especially when odds-ratio calculations require second-order derivatives. Faster samplers, distillation, or fewer-step diffusion methods could make rare-event likelihood estimation more scalable. Better density estimation and more stable log-odds computations would improve confidence in downstream probability estimates. More precise guidance functions could also target different event types, intensities, and regions while preserving physical consistency.

Tropical cyclones are one example of extreme events. Similar methods could be explored for atmospheric rivers, heat waves, severe storms, drought conditions, and compound extremes. Guided probability estimates may also support new attribution-like experiments by comparing event likelihoods across different boundary conditions or climate states, when the diffusion emulator is trained on such different climates.

## Get started

Read the paper, [Towards accurate extreme event likelihoods from diffusion model climate emulators](https://arxiv.org/abs/2605.03802), and explore the Earth2Studio example, `[05_cbottle_tc_guidance.py](https://github.com/NVIDIA/earth2studio/blob/main/examples/07_misc/05_cbottle_tc_guidance.py)`.

For more background, see the cBottle paper, [Climate in a Bottle](https://arxiv.org/abs/2505.06474v1), and the [NVIDIA Earth2Studio GitHub repository](https://github.com/NVIDIA/earth2studio).