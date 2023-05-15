#clean environment
rm(list = ls())
gc()
library("devtools")
library("ape")
#load SymSim2
load_all("SymSim2")


###set gene number, atac region number and cell number
##we used following params to generate data sets
ngenes_list <- c(5000, 8000, 10000, 8000, 10000)
nregions_list <- c(18000, 20000, 22000, 20000, 22000)
ncells_totals <- c(5000, 8000, 10000, 8000, 10000)
min_popsizes <- c(500, 800, 1000, 800, 1000)
###Define trajectory structure
#following are three structure we used for data sets
phyla1 <- read.tree(text="(t1:4.5,t2:1.5,t3:0.5,t4:2,t5:2.5);")
phyla2 <- read.tree(text="(t1:4.5,t2:1.5,t3:0.5,t4:2,t5:2.5,t6:9,t7:6.5,t8:6);")
phyla3 <- read.tree(text="(t1:4.5,t2:1.5,t3:0.5,t4:2,t5:2.5,t6:9,t7:6.5,t8:6,t9:6,t10:7.5,t11:10,t12:11);")
phylas <- list(phyla1, phyla2, phyla3, phyla2, phyla3)
###alpha_mean control drop out rate of RNA
alpha_means <- c(0.00075, 0.00075, 0.00075, 0.0005, 0.0003)
###target prop control drop out rate of ATAC
target_props <- c(0.1, 0.1, 0.1, 0.12, 0.15)

seed <- 123
# the probability that a gene is regulated by respectively 0, 1, 2 regions
p0 <- 0.01
prob_REperGene <- c(p0, (1-p0)*(1/10), (1-p0)*(1/10),(1-p0)*(1/5), (1-p0)*(1/5),(1-p0)*(1/5),(1-p0)*(1/10),(1-p0)*(1/10))
cumsum_prob <- cumsum(prob_REperGene)

###generate five data sets
for(index in 1 : 5){
  ngenes <- ngenes_list[index]
  nregions <- nregions_list[index]
  ncells_total <- ncells_totals[index]
  min_popsize <- min_popsizes[index]
  phyla <- phylas[[index]]
  alpha_mean <- alpha_means[index]
  target_prop <- target_props[index]
  region2gene <- matrix(0, nregions, ngenes)
  set.seed(seed)
  rand_vec <- runif(ngenes)
  
  for (igene in 1:ngenes){
    if (rand_vec[igene] >= cumsum_prob[1] & rand_vec[igene] < cumsum_prob[2]) {
      region2gene[round(runif(1,min = 1, max = nregions)),igene] <- 1 
    } else if (rand_vec[igene] >= cumsum_prob[2]){
      startpos <- round(runif(1,min = 1, max = nregions-1))
      region2gene[startpos: (startpos+1),igene] <- c(1,1)
    }
  }
  
  
  
  
  
  
  
  
  ########################################################################
  #
  # Simulate true scATAC-Seq and scRNA-Seq
  #
  ########################################################################
  # simulate the true count, the parameter setting the same as symsim
  true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total,
                                        min_popsize=min_popsize,
                                        i_minpop=2,
                                        ngenes=dim(region2gene)[2], 
                                        nregions=dim(region2gene)[1],
                                        region2gene=region2gene,
                                        atac_effect=0.8,
                                        evf_center=1,evf_type="continuous",nevf=12,
                                        n_de_evf=8,n_de_evf_atac = 3, impulse=F,vary='s',Sigma=0.7,
                                        phyla=phyla,geffect_mean=0,gene_effects_sd=1,gene_effect_prob=0.3,
                                        bimod=0,param_realdata="zeisel.imputed",scale_s=0.8,
                                        prop_hge=0.015, mean_hge=5, randseed=seed, gene_module_prop=0)
  
  atacseq_data <- true_counts_res[[2]]
  rnaseq_data <- true_counts_res[[1]]
  
  
  ###Simulate technical noise
  
  data(gene_len_pool)
  gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
  save(gene_len, file=sprintf("../NewSym_med_gene_len%d.rda", index))
  
  observed_rnaseq <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[4]], 
                                         protocol="UMI", alpha_mean=alpha_mean, alpha_sd=0.05, 
                                         gene_len=gene_len, depth_mean=5e5, depth_sd=3e3)
  ###generated observed scATAC-Seq count
  atacseq_data <- round(atacseq_data)
  atacseq_noisy <- atacseq_data
  for (icell in 1:ncells_total){
    for (iregion in 1:nregions){
      if (atacseq_data[iregion, icell]>0){
        atacseq_noisy[iregion, icell] <- rbinom(n=1, size = atacseq_data[iregion, icell], prob = 0.3)}
      if (atacseq_noisy[iregion, icell] > 0){
        atacseq_noisy[iregion, icell] <- atacseq_noisy[iregion, icell]+rnorm(1, mean = 0, sd=atacseq_noisy[iregion, icell]/10)
      }
    }
  }
  atacseq_noisy[atacseq_noisy<0.1] <- 0
  prop_1 <- sum(atacseq_noisy>0.1)/(dim(atacseq_noisy)[1]*dim(atacseq_noisy)[2])
  target_prop_1 <- target_prop
  if (prop_1 > target_prop_1) { # need to set larger threshold to have more non-zero values become 0s
    n2set0 <- ceiling((prop_1 - target_prop_1)*dim(atacseq_data)[1]*dim(atacseq_data)[2])
    threshold <- sort(atacseq_noisy[atacseq_noisy>0.1])[n2set0]
    atacseq_noisy[atacseq_noisy<threshold] <- 0
  } else {
    print(sprintf("The proportion of 1s is %4.3f", prop_1))
  }
  
  ###Save the result
  
  datapath <- sprintf("./seed_%d", seed)
  if(index == 1){
    system(sprintf("mkdir %s", datapath))
  }
  write.table(region2gene, sprintf("%s/NewSym_med_region2gene%d.txt", datapath, index),
              quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(observed_rnaseq[[1]], sprintf("%s/NewSym_med_RNA%d.txt", datapath, index), 
              quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(atacseq_noisy, sprintf("%s/NewSym_med_ATAC%d.txt", datapath, index), 
              quote=F, row.names = F, col.names = F, sep = "\t")
  write.table(true_counts_res[[4]][,1:2], sprintf("%s/NewSym_med_label%d.txt", datapath, index), 
              quote=F, row.names = F, col.names = T, sep = "\t")

}


