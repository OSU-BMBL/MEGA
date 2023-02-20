setwd("/bmbl_data/cankun_notebook/daniel/pyMEGA/figures/data")
library(tidyverse)
library(taxizedb)


# Set dataset file path
metadata_file <- "cre_metadata_full.csv"
phy_file <- "cre_abundance_data_phy_matrix.csv"
metabolic_file <- "cre_abundance_data_metabolic_matrix.csv"
taxa_num_file <- "cre_abundance_data_taxa_num.csv"
final_taxa_file <- "cre_abundance_data_final_taxa.txt"


# Load dataset
this_tax_num <- read.csv(taxa_num_file)
this_final <- readLines(final_taxa_file)
this_final <- str_split(this_final, "\t")
this_name_final <- lapply(this_final, function(x) {
  tmp <- na.omit(as.integer(x))[-1]
  return (taxid2name(tmp))
})


# Convert final taxa to list
label_res <- read.csv(metadata_file) %>%
  dplyr::select(2:3) %>%
  unique() %>%
  dplyr::pull(cancer_name)
names(this_name_final) <- label_res

id_res <- lapply(this_final, function(x){
  tmp <- na.omit(as.integer(x))[-1]
  return (tmp)
})

names(id_res) <- label_res
saveRDS(id_res, "id_result.rds")

name_res <- this_name_final
saveRDS(name_res, "name_result.rds")


###### Save phylogenetics relation in names
phy_df <- read.csv(phy_file, header = T)
all_species_name <- taxizedb::taxid2name(phy_df$X)
phy_df$X <- all_species_name

dup_id <- which(duplicated(all_species_name))
phy_df <- phy_df[-dup_id,]
na_row <- which(is.na(phy_df$X))
phy_df$X[na_row] <- "Empty"
rownames(phy_df) <- phy_df$X
phy_df <- phy_df[,c(-1, -dup_id)]
colnames(phy_df) <- rownames(phy_df)

write.table(phy_df, gsub("\\.csv", "_name.csv", phy_file), sep=",", quote = F, col.names = T, row.names = T)

df <- stack(name_res) %>%
  dplyr::select(2,1) %>%
  rename(`TF`=`ind`, `enhancer`=`values`) %>%
  mutate(gene = TF)

df_meta <- read.csv(metadata_file) %>%
  dplyr::select(cancer_name) %>%
  group_by(cancer_name) %>%
  dplyr::count()

weight_df <- this_tax_num
colnames(weight_df) <- c("species", names(name_res))
weight_df$species <-  taxizedb::taxid2name(as.integer(weight_df$species))

i="COAD"
for (i in df_meta$cancer_name) {
  this_total <- df_meta%>%
    dplyr::filter(cancer_name == i) %>%
    pull(n)
  weight_df[,i] <- as.numeric(weight_df[,i]) / as.numeric(this_total)
}

write.table(weight_df, "weight_df.csv", sep=",", quote = F, col.names = T, row.names = T)


###### Save metabolic relation
metabolic_df <- read.csv(metabolic_file, header = T)
all_species_name <- taxizedb::taxid2name(metabolic_df$X)
metabolic_df$X <- all_species_name
colnames(metabolic_df) <- c("X", all_species_name)
write.table(metabolic_df, gsub("\\.csv", "_name.csv", metabolic_file), sep=",", quote = F, col.names = T, row.names = F)
