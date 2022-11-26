# Uczenie Nienadzorowane - projekt


## Generator danych
...


## Wykorzystywane metody:

1) Autoenkoder:
   - [ ] Autoenkodery odszumiające (ang. denoising autoencoders) DAE    Rafał, 
   - [ ] Autoenkodery rzadkie (ang. sparse autoencoders) SAE            Bartek
   - [ ] Autoenkodery wariacyjne (ang. variational autoencoders) VAE    Kasia


2) reprezentacji i wizualizacji przestrzeni wysokowymiarowych (np. UMAP)


3) klasyczne metody klasteryzacji (np. spektralne / kmeans),


4) maping


5) jezeli nie wychodzi->

   neuronowe nienadzorowane metody redukcji wymiarowosci: (/dimension_reduction/main.ipynb)
   - [x] Principal component analysis (PCA)
   - [x] Kernel Principal component analysis (KPCA)
   - [x] Sparse Principal Components Analysis (SparsePCA)
   - [x] Fast ICA: a fast algorithm for Independent Component Analysis (ICA)
   - [x] Non-Negative Matrix Factorization (NMF)
   - [x] Isomap Embedding (Isomap)
   - [x] Multidimensional scaling (MDS)
   - [x] Locally Linear Embedding (LLE) 
   - [x] Laplacian Eigenmaps (LEM) Spectral embedding for non-linear dimensionality reduction
   


autoenkodery [Link](https://miroslawmamczur.pl/czym-sa-autoenkodery-autokodery-i-jakie-maja-zastosowanie/)

Plan dzialan:
- Autoenkoder 
- umap -> duzo kulek
- jezeli da sie podzielic na klastry -> do klasteryzatora 
- jezeli klastrw ~47 (45 - 49) -> mapowanie po ilosci populacji
- jezeli klastrow mniej -> sprobowac klasteryzacje z innymi parametrami(kmean / spectral )
- if umap = 1/2 -> nauczyc inaczej autoenkoder
- jezeli nie wychodzi -> sprobowac z redukcj wymiarowosci
