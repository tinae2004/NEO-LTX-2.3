# NEO FastFrame Vast GPU training

This is a manually triggered, budget-capped first GPU training run. It searches verified on-demand RTX 3090 / RTX 4090 offers with at least 24 GB VRAM and at least 99% listed reliability. It rents only an offer within both price limits, reserving 25% of the stated total budget.

## Required repository secret

Add a Vast.ai API key as the GitHub Actions repository secret named VAST_API_KEY. The key stays on the GitHub Actions runner and is never copied to the rented machine.

## Run

Open Actions, select NEO FastFrame - Vast GPU Training, and choose Run workflow. Enter the total USD budget, maximum hourly price and maximum rental duration. The workflow refuses to rent if no eligible RTX offer fits those limits.

The runner registers an ephemeral SSH key, creates an on-demand instance, waits for the instance and NVIDIA driver, transfers the checked-out commit, generates synthetic and Blender motion samples, builds dense geometry-flow labels, trains the current skeleton-control baseline on CUDA, downloads its best/latest checkpoints and metrics, then sends a Vast destroy request in a finally block. Outputs are uploaded as a GitHub Actions artifact.

## Scope and limitations

This run trains the existing 17-joint skeleton-to-2D-control baseline. It does not train a dense RGB optical-flow network, a FLUX adapter, or a production video model. Blender geometry-derived dense flow labels are generated and retained as a separate research dataset; the current skeleton trainer does not consume them. Training uses generated synthetic labels plus a small three-action Blender capture. A successful run proves GPU execution, not photorealistic video quality or generalization.

The instance is on-demand, not interruptible. The workflow has a hard runtime limit and destroys the instance after collecting outputs. If the GitHub runner itself is forcibly terminated, check the Vast console for a leftover instance before launching another run.
