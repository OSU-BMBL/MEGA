if(!require('taxizedb')) {
  install.packages('taxizedb', repos='http://cran.us.r-project.org')
  library('taxizedb')
}
options (warn = -1)
Args <- commandArgs()

data <- read.table(Args[6], header=F, sep="\n")
taxa <- as.data.frame(matrix(1:(dim(data[1])*2), nrow = dim(data[1]), ncol = 2))

for (i in 1:dim(data)[1]){
    a = taxa_at(data[i,1], rank = "genus", db="ncbi", missing = "lower",verbose = FALSE, warn = FALSE)[1]
    b = as.data.frame(a)
    if (dim(b)[2]==3)
        taxa[i,1] = b[1,3]
    else   
        taxa[i,1] = 0
    taxa[i,2] = data[i,1]
}

file_name <- strsplit(basename(Args[6]),split="\\.")[[1]][1]
file_a = paste(c(dirname(Args[6]),'/',file_name,'_phy_relation.csv'),collapse = "")
write.table(taxa, file_a, sep='\t', row.names = FALSE, col.names = FALSE, quote = F)
# print (paste('The final metabolic relations among species and species is in', file_a))