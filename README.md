# README

## Manuscript code: gradients, eccentricity, FC, statistics & surface plots

Analysis and figure-generation scripts used in the manuscript. Includes pipelines for functional connectivity (FC), gradient mapping, eccentricity, inferential stats, and publication-quality figures.

### What’s inside
- Deriving covariance/FC matrices; VAN/SM network extraction  
- Riemannian centering and PCA gradient mapping  
- Eccentricity computation & visualization (3D and cortical surfaces)  
- rmANOVA per ROI; per-ROI correlations with FDR correction  
- Seed-based FC extraction & pairwise contrasts with FDR  
- RSA model testing across time/epoch/day  
- Clustering + validation (Silhouette, Davies–Bouldin)  
- UMAP, radar/spider plots, and final figure generation

All scripts are CLI tools with self-documented `--help`.

---

## Requirements
- Python ≥ 3.10
- Recommended: create a virtualenv/conda env
- See `requirements.txt` for exact versions

```bash
pip install -r requirements.txt
```


