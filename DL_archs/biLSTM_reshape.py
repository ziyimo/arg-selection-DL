## Reshaping for bidirectional LSTM ##
def reshape(fea_df): 
  ## Scale by sample size (no. of chromosome) ##
  fea_df = fea_df/200
  fea_reshaped = fea_df.reshape((-1, 13, 300))
  return fea_reshaped

def plot_example(reshaped_df, meta_df):
  idx = np.random.randint(reshaped_df.shape[0])
  print(idx, meta_df[idx])
  plt.matshow(reshaped_df[idx], cmap="Greens")