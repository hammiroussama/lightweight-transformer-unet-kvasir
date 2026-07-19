# Package status

## Completed and checked

- Four model configurations implemented.
- Unit tests for model output shape and metrics pass.
- A synthetic one-epoch train/evaluate smoke test completed successfully.
- Deterministic split generation and worker seeding implemented.
- Test metrics, HD95, parameter count, latency, hardware, and software versions are logged.
- Training curves, Dice comparison, qualitative best/worst cases, and LaTeX macros are generated automatically.
- Experimental LaTeX paper compiles with all citations resolved.
- The architecture figure was visually inspected after PDF rendering.
- GitHub URL in README, paper, and CITATION.cff:
  https://github.com/hammiroussama/lightweight-transformer-unet-kvasir

## Still required from the researcher

- Download Kvasir-SEG from the official dataset page.
- Run the four full experiments on the real dataset.
- Confirm `paper/results_macros.tex` no longer contains `TBD`.
- Inspect generated qualitative results and write model-specific observations.
- Replace author email and supervisor placeholders.
- Upload the package and generated results to the public GitHub repository.
- Recompile the final paper after real metrics and figures are generated.

No experimental metric has been fabricated in this package.
