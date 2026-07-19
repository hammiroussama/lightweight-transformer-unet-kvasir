# Upload this package to the existing GitHub repository

Repository:

```text
https://github.com/hammiroussama/lightweight-transformer-unet-kvasir
```

## Option A: replace an empty repository from the command line

Extract this package, open a terminal inside the extracted folder, and run:

```bash
git init
git branch -M main
git remote add origin https://github.com/hammiroussama/lightweight-transformer-unet-kvasir.git
git add .
git commit -m "Add reproducible Kvasir-SEG experimental pipeline"
git push -u origin main
```

When `origin` already exists, use:

```bash
git remote set-url origin https://github.com/hammiroussama/lightweight-transformer-unet-kvasir.git
git add .
git commit -m "Add reproducible Kvasir-SEG experimental pipeline"
git push -u origin main
```

## Option B: upload using the GitHub website

1. Open the repository.
2. Choose **Add file** then **Upload files**.
3. Upload the extracted files and folders, not the outer ZIP itself.
4. Commit to the `main` branch.

## After experiments

Commit the fixed split, result tables, training histories, generated figures, and the final paper source. Do not commit the Kvasir-SEG dataset or large checkpoints unless Git LFS is configured and redistribution is permitted.
