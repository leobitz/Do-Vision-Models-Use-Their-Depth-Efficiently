# Do vision models use their depth efficiently?

This repository contains five notebook experiments on layer contributions,
layer approximation, component noise, direct logit attribution, linear
probing in DeiT models.

## Run on Modal

The Modal runner executes one model per remote job on a T4 GPU. Each successful
job writes its CSV files, executed notebook, and environment manifest to a
persistent Modal volume. A repeated command skips jobs that already have a
success marker.

Run a small pilot before starting the full suite:

```bash
modal run modal_runner.py --experiment all --model "DeiT-tiny" --pilot
```

Run every experiment and model:

```bash
modal deploy modal_runner.py
python3 submit_modal_jobs.py submit --experiment all --model all
python3 submit_modal_jobs.py wait
```

Run one experiment or model by using its name. Comma-separated selections are
also accepted:

```bash
python3 submit_modal_jobs.py submit \
  --experiment exp3_component_noise \
  --model "DeiT-base distilled 384"
```

Download the persistent results after the jobs finish:

```bash
modal volume get vision-depth-results / cloud_results
python3 collect_results.py cloud_results
```

The runner uses `vision-depth-cache` for Hugging Face downloads and
`vision-depth-results` for outputs. It never writes experiment outputs into the
source checkout.

The deployed submission path is preferred for full runs. Modal owns the queued
calls, so closing the terminal or losing a local network connection does not
cancel remote work. The function allows one container, which keeps GPU use at
one T4 at a time.
