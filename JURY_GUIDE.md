# Jury Presentation Guide

## 30-second pitch
"Factory cameras generate gigabytes of part images daily, but edge inspection devices have tiny disks and git cannot version binaries. We separate concerns: DVC versions the image datasets in a MinIO object store while git versions only tiny pointer files, MLflow records exactly which data and hyperparameters produced each model, and the final model ships inside a small Docker container exposing a FastAPI endpoint that flags defective parts in under 3 milliseconds on CPU."

## Likely jury questions and answers

**Q: Why not just store images in git?**
- Git copies every version of every file into `.git`; a 5 GB image set with 10 versions bloats the repo toward 50 GB
- Git diffs binaries poorly and clones become unusable
- DVC stores one content-addressed copy per unique file in S3/MinIO and git holds a 200-byte `.dvc` file with the hash. Checkout any commit, run `dvc pull`, and you get the exact dataset of that day

**Q: Why an autoencoder instead of a classifier (like ResNet)?**
- A classifier needs labeled defect examples; real lines have few defects and new defect types appear that were never in training
- The autoencoder trains on good parts only and flags ANYTHING abnormal, including never-seen defect types
- It is also 100x smaller than ResNet, which matters on edge CPUs

**Q: How is the defect threshold chosen?**
- After training, we score all good training images and set threshold = mean + 3 standard deviations
- Statistically, ~99.7% of good parts fall below it, so false rejects are rare
- The threshold is saved inside the model checkpoint and logged to MLflow, so it travels with the model

**Q: Why patch-based scoring instead of whole-image error?**
- A 3-pixel-wide scratch changes the global average error of 16,384 pixels almost not at all
- We average error in 8x8 patches and take the worst patch; a small local defect dominates its patch and gets caught
- In our benchmark this raised defect recall from 21/30 to 49/50

**Q: What does MLflow add over print statements?**
- Every run stores hyperparameters, per-epoch loss, threshold, model artifact, and a tag with the git commit (which pins the DVC data version)
- Full lineage: "the model on device 7 came from data version abc123 with lr=0.001"
- The registry supports promoting models through staging to production

**Q: What exactly runs on the edge device?**
- Only the Docker container: FastAPI + CPU PyTorch + OpenCV + the 2.8 MB model
- No dataset, no DVC, no MLflow, no network dependency at inference time
- Multi-stage build keeps the image lean; runs as a non-root user with a healthcheck

**Q: How do you handle lighting changes on the factory floor?**
- CLAHE (adaptive histogram equalization) in preprocessing normalizes local contrast
- Gaussian blur removes sensor noise so it is not mistaken for texture defects
- The same preprocess function is imported by both training and serving, eliminating train/serve skew

**Q: How does a new day's images get into the system?**
- `dvc pull` on the training server, copy new images in, `dvc add`, `git commit`, `dvc push`
- `dvc repro` re-runs training only if data or code hashes changed (build-system-like caching)
- New container is built with the new model and rolled out to devices

**Q: Measured performance?**
- 50/50 good parts passed, 49/50 injected defects caught (scratches, holes, edge chips)
- ~2.6 ms per image on 2 CPU threads, ~695K parameters
- Heatmap endpoint localizes the defect for the operator

## Demo script (5 minutes)
1. `docker compose up -d` then show MinIO console (9001) and MLflow UI (5000)
2. Show `data/raw.dvc` in git: "this 4-line file IS the dataset in git's eyes"
3. `curl -F file=@good.png localhost:8000/inspect` -> PASS
4. `curl -F "file=@defect.png" "localhost:8000/inspect?heatmap=true"` -> DEFECT, decode and show heatmap
5. Open MLflow run: point at logged threshold and data_version tag
