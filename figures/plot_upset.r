# Please change the working directory
setwd("/bmbl_data/cankun_notebook/daniel/code/mega_test/RA_norm_e50_gpu0_v1")
library(ComplexHeatmap)
library(ggplot2)


name_res <- readRDS("name_result.rds")

upset_df <- list_to_matrix(name_res)
m = make_comb_mat(name_res)
p1 <- UpSet(m)


pdf(
  paste0("fig2b_upset.pdf"),
  width = 8,
  height = 5
)
print(p1)
dev.off()

