## Reshaping for CNN_1d3c, time-wise covolution ##
def reshape(fea_df): 
  fea_reshaped = np.concatenate((fea_df[:, 0::3, :], fea_df[:, 1::3, :], fea_df[:, 2::3, :]), axis=1)
  # Add a dimension for Channel
  fea_reshaped = np.expand_dims(fea_reshaped, fea_reshaped.ndim)
  return fea_reshaped

def plot_example(reshaped_df, meta_df):
  idx = np.random.randint(reshaped_df.shape[0])
  print(idx, meta_df[idx])
  plt.matshow(reshaped_df[idx].reshape(reshaped_df.shape[1:3]))