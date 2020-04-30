library(gplots)
lineage_matrix <- read.csv("/Volumes/GoogleDrive/My\ Drive/python_code/connectome_tools/lineage_analysis/lineage_matrix.csv")

palette <- colorRampPalette(c("white", "orange", "brown4","brown4","brown4","brown4","brown4","brown4"))(n = 128)
width <- 10
height <- 10
pointsize <- 4

#setEPS()
#postscript("/Volumes/GoogleDrive/My\ Drive/R\ code/CNpaper/new_names_analysis/plots/ExtFig4d_MB2ONsimilarity_by_MBON-input_CNnames_cluster.eps", width = width, height = height, pointsize = pointsize)
heatmap.2(as.matrix(lineage_matrix), scale="none", density.info="none", trace="none", col=palette, key.xlab="Similarity", key.title=FALSE, keysize = 1, key = FALSE)
#dev.off()



