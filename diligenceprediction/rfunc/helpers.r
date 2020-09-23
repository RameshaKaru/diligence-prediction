library(kde1d)

func_draw_unconstrained_kde <- function(percentages){
  # Draw unconstrained KDE
  #
  # Args:
  #   percentages: vector of percentages.
  #
  # Returns:
  #   KDE.
  fit <- kde1d(percentages) # estimate density
  summary(fit)              # information about the estimate

  # uncomment below to output the plot
  #plot(fit, lwd=0.5, main = "Unconstrained KDE", xlab = "Percentages") # plot the density estimate

  return(fit)

}

func_plot_final_kde <- function(fit, name){
  # Plot final KDE
  #
  # Args:
  #   fit: final KDE.
  #   name: plot name.

  jpeg(paste('diligenceprediction/rfunc/kdeplots/',name,'_KDE.jpg', sep=""))
  plot(fit, lwd=0.5, main = paste(name," KDE"), xlab = "Percentages") # plot the density estimate
  dev.off()

}

func_get_prob_mass_trans <- function(fit_bounded, a, b){
  # get the area under the curve in a given range.
  #
  # Args:
  #   fit_bounded: KDE.
  #   a: lower value.
  #   b: upper value.
  #
  # Returns:
  #   area under the curve between a and b.
  auc_val <- pkde1d(b,fit_bounded) - pkde1d(a,fit_bounded)

  #print(paste("AUC from",a,"to",b,":",auc_val))

  return(auc_val)
}


func_modify_data <- function(percentages, low_cut, up_cut,low,up){
  # cut of extreme data at ends (if any)
  #
  # Args:
  #   percentages: unmodified percentages
  #   low_cut: where to cut if no peak at lower extreme
  #   up_cut: where to cut if no peak at upper extreme
  #   low: lower bound.
  #   up: upper bound.
  #
  # Returns:
  #   percentages after removing extremes.

  tot_data <- length(percentages)
  count_0 <- sum(abs(percentages - low) < 1e-6)
  count_100 <- sum(abs(percentages - up) < 1e-6)

  print(paste("Data percentage at",low,count_0/tot_data*100))
  print(paste("Data percentage at",up,count_100/tot_data*100))

  # cut off peaks at 0
  if(count_0/tot_data > 0.01){
    next_min <- min( percentages[percentages!=min(percentages)] )
    percentages <- subset(percentages, percentages>next_min)
    print(paste('Cutting values below',next_min))

  # no peak, cut off at given low_cut
  }else{
    percentages <- subset(percentages, percentages>low_cut)
    print(paste('Cutting values below',low_cut))
  }

  # cut off peaks at 100
  if(count_100/tot_data > 0.01){
    next_max <- max( percentages[percentages!=max(percentages)] )
    percentages <- subset(percentages, percentages<next_max)
    print(paste('Cutting values above',next_max))

  # no peak, cut off at given up_cut
  }else{
    percentages <- subset(percentages, percentages<up_cut)
    print(paste('Cutting values above',up_cut))
  }

  return(percentages)

}

func_cut_off_clusters <- function(raw_percentages, low_cut, up_cut, low=0, up=100){
  # draws new KDE after removing extreme points.
  #
  # Args:
  #   raw_percentages: unmodified percentages
  #   low_cut: where to cut if no peak at lower extreme
  #   up_cut: where to cut if no peak at upper extreme
  #   low: lower bound.
  #   up: upper bound.
  #
  # Returns:
  #   modified KDE.
  fit <- func_draw_unconstrained_kde(raw_percentages)
  percentages <- func_modify_data(raw_percentages, low_cut, up_cut, low, up)
  new_fit <- func_draw_unconstrained_kde(percentages)
  lower_bound <- floor(qkde1d(0,new_fit))
  upper_bound <- ceiling(qkde1d(1,new_fit))
  print(paste('Lower bound', lower_bound))
  print(paste('Upper bound', upper_bound))
  print(paste('AUC between lower bound and',low, pkde1d(low,new_fit)))
  print(paste('AUC between upper bound and',up, pkde1d(upper_bound,new_fit) - pkde1d(up,new_fit)))
  print("Unconstrained KDE after cutting off peaks")
  return(new_fit)
}
