
library("devtools")
library("ape")
load_all("../")
seed <- 123
###Params like RNA and ATAC
ncells_totals <- c(5000, 8000, 10000, 8000, 10000)
###Define trajectory structure
#following are three structure we used for data sets
phyla1 <- read.tree(text="(t1:4.5,t2:1.5,t3:0.5,t4:2,t5:2.5);")
phyla2 <- read.tree(text="(t1:4.5,t2:1.5,t3:0.5,t4:2,t5:2.5,t6:9,t7:6.5,t8:6);")
phyla3 <- read.tree(text="(t1:4.5,t2:1.5,t3:0.5,t4:2,t5:2.5,t6:9,t7:6.5,t8:6,t9:6,t10:7.5,t11:10,t12:11);")
phylas <- list(phyla1, phyla2, phyla3, phyla2, phyla3)
###alpha_mean control drop out rate of Protein
alpha_means <- c(0.045, 0.045, 0.045, 0.035, 0.025)

for(index in 1 : 5){
  alpha_mean <- alpha_means[index]
  phyla <- phylas[[index]]
  ncells_total <- ncells_totals[index]
  true_ADTcounts_res <- SimulateTrueCounts(ncells_total=ncells_total, 
                                           min_popsize=50, 
                                           i_minpop=1, 
                                           ngenes=1250, 
                                           nevf=10, 
                                           evf_type="discrete", 
                                           n_de_evf=6, 
                                           vary="s", 
                                           Sigma=0.3, 
                                           phyla=phyla,
                                           param_realdata="protein_sim",
                                           randseed=seed)
  ###load gene_len from RNA random result for better mapping of protein to RNA
  load(sprintf("../NewSym_med_gene_len%d.rda", index))
  observed_ADTcounts <- True2ObservedCounts(true_counts=true_ADTcounts_res[[1]], 
                                            meta_cell=true_ADTcounts_res[[3]], 
                                            protocol="UMI", 
                                            alpha_mean=0.045, 
                                            alpha_sd=0.01, 
                                            gene_len=gene_len, 
                                            depth_mean=50000, 
                                            depth_sd=3000,
  )
  counts2 <- observed_ADTcounts[[1]]
  
  #filter
  rownames(counts2) <- paste("G",1:nrow(counts2),sep = "")
  colnames(counts2) <- paste("C",1:ncol(counts2),sep = "")
  library(Seurat)
  pbmc <- CreateSeuratObject(counts = counts2, project = "P2", min.cells = 0, min.features = 0)
  pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize")
  pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 250)
  counts2 <- counts2[pbmc@assays[["RNA"]]@var.features,]
  datapath <- sprintf("../../Symsim2/seed_%d", seed)
  write.table(counts2, sprintf("%s/NewSym_med_Protein%d.txt", datapath, index), 
              quote=F, row.names = F, col.names = F, sep = "\t")
}
