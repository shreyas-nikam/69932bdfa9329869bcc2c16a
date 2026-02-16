
# Synthetic Time-Series Generation for Enhanced Backtesting & Stress Testing

## Navigating Uncertainty: Augmenting Financial Data with Generative Models

**Persona:** Sarah, a Quantitative Strategist at "Alpha Horizons Investment Management."

**Scenario:** Sarah's firm relies heavily on backtesting strategies and stress testing portfolios. However, she constantly faces the "single history problem": rare but impactful market events (like the 2008 GFC or 2020 COVID crash) have only occurred once or twice in historical data. This scarcity limits the robustness of her analyses, making it difficult to assess how strategies truly perform under diverse, unseen, yet plausible market conditions. Traditional Monte Carlo simulations, based on simple parametric models like Geometric Brownian Motion (GBM), often fail to capture the complex, non-Gaussian statistical properties of real financial data—such as fat tails, volatility clustering, and autocorrelation.

**Challenge:** Sarah needs a way to generate new, realistic financial price paths that faithfully mimic the complex statistical characteristics of historical data, going beyond simple parametric assumptions. These synthetic paths will augment the historical record, allowing her to conduct more comprehensive backtesting and stress testing, ultimately leading to more robust strategy development and risk management.

**Goal of this Notebook:** This notebook will guide Sarah through a practical workflow to generate realistic synthetic financial time series using a **TimeGAN (Generative Adversarial Network for time series)**. She will compare the synthetic data's quality against both real historical data and a traditional **Geometric Brownian Motion (GBM)** baseline. Finally, she will apply a simple investment strategy to these synthetic paths to assess its robustness across a multitude of plausible market scenarios.

---

## 1. Setting Up the Environment and Acquiring Historical Data

Sarah begins by setting up her Python environment and acquiring the necessary historical financial data. For robust generative modeling, she needs multi-feature (OHLCV) daily data for a representative market index.

### 1.1. Install Required Libraries

Before proceeding, Sarah ensures all necessary Python libraries are installed.

```python
!pip install yfinance pandas numpy scipy matplotlib scikit-learn tensorflow ydata-synthetic
```

### 1.2. Import Dependencies

Next, Sarah imports the core libraries for data manipulation, financial data retrieval, statistical analysis, plotting, and deep learning.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, describe
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, LSTM)
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Try importing TimeGAN from ydata-synthetic
try:
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    from ydata_synthetic.synthesizers.timeseries import ModelParameters
    TIMEGAN_AVAILABLE = True
except ImportError:
    print("ydata-synthetic not installed or failed to import; using custom GAN implementation.")
    TIMEGAN_AVAILABLE = False

print(f"TensorFlow Version: {tf.__version__}")
```

### 1.3. Acquire and Preprocess Financial Data

Sarah downloads daily OHLCV data for the S&P 500 ETF (SPY) and transforms it into daily percentage returns and log-transformed volume changes. This preprocessing is crucial for modeling financial time series, as returns are more stationary than prices.

The percentage change for prices is calculated as $\frac{P_t - P_{t-1}}{P_{t-1}}$, and for volume, a log-difference transformation $V_t = \ln(1+V_t) - \ln(1+V_{t-1})$ is often used to stabilize variance.

```python
def load_and_preprocess_data(ticker='SPY', start_date='2010-01-01', end_date='2024-01-01'):
    """
    Loads historical OHLCV data, calculates percentage changes for prices,
    and log-transformed differences for volume.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker and date range.")

    features_df = pd.DataFrame({
        'open': data['Open'].pct_change(),
        'high': data['High'].pct_change(),
        'low': data['Low'].pct_change(),
        'close': data['Close'].pct_change(),
        'volume': np.log1p(data['Volume']).diff() # log(1+x) then diff for volume
    })

    # Drop any rows with NaN values resulting from pct_change/diff
    features_df = features_df.dropna()
    
    print(f"\nTraining data shape: {features_df.shape[0]} days x {features_df.shape[1]} features")
    close_returns = features_df['close']
    print(f"Close return stats: mean={close_returns.mean():.5f}, std={close_returns.std():.4f}, "
          f"skew={close_returns.skew():.2f}, kurt={close_returns.kurtosis():.2f}")
    
    return features_df

# Execute data loading and preprocessing
features = load_and_preprocess_data(ticker='SPY', start_date='2010-01-01', end_date='2024-01-01')
```

### 1.4. Explanation of Data Preprocessing

Sarah knows that financial time series, especially raw prices, are typically non-stationary. Working with returns stabilizes the data, making it more suitable for modeling. The `dropna()` step removes initial `NaN` values created by the percentage change calculations, ensuring a clean dataset for training. The descriptive statistics for 'close' returns provide initial insights into the data's characteristics, especially its mean, standard deviation, skewness, and kurtosis, which are crucial for later comparisons with synthetic data.

---

## 2. Baseline Model: Geometric Brownian Motion (GBM) Simulation

Before diving into complex generative models, Sarah wants to establish a traditional benchmark. The **Geometric Brownian Motion (GBM)** model is a fundamental stochastic process widely used in finance to model asset prices. It assumes that asset returns are normally distributed and exhibit constant drift and volatility.

The stochastic differential equation for GBM is given by:
$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
where $S_t$ is the asset price at time $t$, $\mu$ is the drift coefficient (expected return), $\sigma$ is the volatility coefficient, and $dW_t$ is a Wiener process (standard Brownian motion).

Its discrete-time solution for simulating price paths is:
$$ S_{t+1} = S_t \exp \left[ \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} Z_t \right], \quad Z_t \sim \mathcal{N}(0,1) $$
Here, $\Delta t$ is the time step (e.g., $1/252$ for daily steps in a year), and $Z_t$ are independent standard normal random variables.

Sarah will calibrate GBM parameters ($\mu$, $\sigma$) from the historical daily close returns of 'SPY'.

### 2.1. Implement and Calibrate GBM

Sarah implements a function to simulate GBM paths and calibrates its annual drift ($\mu$) and volatility ($\sigma$) using the historical 'close' returns.

```python
def simulate_gbm(mu_annual, sigma_annual, S0, n_days, n_paths, dt_annual=1/252):
    """
    Simulates price paths under Geometric Brownian Motion.
    
    Args:
        mu_annual (float): Annualized drift (mean return).
        sigma_annual (float): Annualized volatility (standard deviation of returns).
        S0 (float): Initial price.
        n_days (int): Number of trading days to simulate for each path.
        n_paths (int): Number of independent price paths to generate.
        dt_annual (float): Time step as a fraction of a year (e.g., 1/252 for daily).
        
    Returns:
        tuple: (returns_paths, price_paths)
            returns_paths (np.array): Simulated daily returns for each path.
            price_paths (np.array): Simulated daily prices for each path.
    """
    # Daily drift and volatility
    mu_daily = mu_annual * dt_annual
    sigma_daily = sigma_annual * np.sqrt(dt_annual)

    # Generate standard normal random variables for each step and path
    Z = np.random.standard_normal((n_paths, n_days))

    # Calculate daily returns using the GBM discrete solution
    # (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t
    daily_returns_factor = np.exp((mu_annual - 0.5 * sigma_annual**2) * dt_annual + sigma_annual * np.sqrt(dt_annual) * Z)
    
    # Initialize prices with S0
    prices = np.full((n_paths, n_days + 1), S0, dtype=float)
    
    # Calculate prices from S0 and daily return factors
    for t in range(n_days):
        prices[:, t+1] = prices[:, t] * daily_returns_factor[:, t]
    
    # Calculate the actual daily returns (for consistency with log returns later)
    returns = np.log(prices[:, 1:] / prices[:, :-1]) # Log returns

    return returns, prices

# Calibrate parameters from historical 'close' returns
historical_close_returns = features['close'].values
mu_annual = np.mean(historical_close_returns) * 252 # Annualize mean
sigma_annual = np.std(historical_close_returns) * np.sqrt(252) # Annualize std
S0 = 100 # Normalized starting price for visualization

print(f"Calibrated GBM: Annual Mean (mu) = {mu_annual:.4f}, Annual Std (sigma) = {sigma_annual:.4f}")

# Generate 1,000 synthetic paths, each 252 days long (approx. 1 trading year)
n_days_sim = 252
n_paths_sim = 1000
gbm_returns, gbm_prices = simulate_gbm(mu_annual, sigma_annual, S0, n_days_sim, n_paths_sim)

print(f"GBM simulated paths shape (returns): {gbm_returns.shape}")
print(f"GBM simulated paths shape (prices): {gbm_prices.shape}")
```

### 2.2. Visualize GBM Simulated Paths

Sarah plots a subset of the generated GBM price paths to visually inspect their behavior.

```python
plt.figure(figsize=(12, 6))
for i in range(20): # Plot 20 sample paths
    plt.plot(gbm_prices[i], alpha=0.5, color='blue')
plt.title('Monte Carlo GBM: 20 Sample Price Paths (S&P 500 ETF)')
plt.xlabel('Trading Day')
plt.ylabel('Price')
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('gbm_sample_paths.png', dpi=150)
plt.show()
```

### 2.3. Limitations of GBM for Financial Data

Sarah recognizes that while GBM is a good starting point, its assumptions are often violated by real financial data. This makes it an insufficient model for robust stress testing and complex strategy backtesting.

*   **No volatility clustering:** GBM assumes constant volatility ($\sigma$). Real markets exhibit periods of high and low volatility (GARCH effects), meaning large price movements tend to be followed by large price movements.
*   **Normal returns:** GBM generates Gaussian (normal) returns. Real financial returns often have "fat tails" (excess kurtosis, where $Kurtosis > 3$) and negative skewness, indicating a higher probability of extreme events and larger downside movements than a normal distribution would predict.
*   **No autocorrelation structure:** GBM returns are independent and identically distributed (i.i.d.). Real returns often show short-term momentum or long-term mean reversion, especially in absolute returns (indicating volatility clustering).
*   **No cross-asset dependence dynamics:** Standard GBM models with constant correlation cannot capture the phenomenon of correlation spikes during crises.

These limitations highlight why Sarah needs a more sophisticated approach like TimeGAN, which can learn and replicate these complex statistical properties directly from data without explicit parametric assumptions.

---

## 3. TimeGAN Model: Training for Realistic Time-Series Generation

To overcome the limitations of GBM, Sarah turns to **TimeGAN (Generative Adversarial Network for time series)**. TimeGAN is specifically designed to capture the complex temporal dependencies and statistical properties inherent in real financial time series. It employs an adversarial training mechanism, where a `Generator` tries to produce synthetic data indistinguishable from real data, and a `Discriminator` tries to distinguish between real and fake. Crucially, TimeGAN adds an embedding network, a recovery network, and a supervised loss to ensure it captures not just marginal distributions, but also temporal dynamics like autocorrelation and volatility clustering.

The overall TimeGAN objective combines three loss components:
$$ \mathcal{L}_{\text{TimeGAN}} = \mathcal{L}_{\text{reconstruction}} + \gamma \mathcal{L}_{\text{supervised}} + \mathcal{L}_{\text{adversarial}} $$
where:
*   $\mathcal{L}_{\text{reconstruction}}$ ensures the autoencoder (embedding and recovery networks) can reconstruct real data from its latent representation.
*   $\mathcal{L}_{\text{supervised}}$ trains the generator to capture one-step-ahead temporal dynamics in the latent space, guiding it to generate sequences that preserve sequential information.
*   $\mathcal{L}_{\text{adversarial}}$ is the standard GAN loss, where the discriminator tries to distinguish real vs. synthetic latent sequences, and the generator tries to fool it.

### 3.1. Prepare Data for TimeGAN Training

TimeGAN requires data to be in sequences (e.g., 20-day windows) and typically normalized. Sarah will create these sequences from her preprocessed OHLCV features and scale them to a range suitable for GAN training (e.g., `[-1, 1]` for `tanh` activation in the generator).

```python
SEQ_LEN = 20 # Length of each historical sequence (e.g., 20 trading days)
N_FEATURES = features.shape[1] # Number of features (OHLCV)
LATENT_DIM = 32 # Dimension of the noise vector for the generator

def prepare_sequences(data, seq_len):
    """
    Creates sliding-window sequences for GAN training.
    """
    sequences = []
    # Ensure there are enough data points to create at least one sequence
    if len(data) < seq_len:
        print("Warning: Not enough data points to create sequences of desired length.")
        return np.array([])
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

# Scale data to [-1, 1] using MinMaxScaler for tanh output
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_features = scaler.fit_transform(features.values)

# Create sequences
real_sequences = prepare_sequences(scaled_features, SEQ_LEN)

print(f"Original features shape: {features.shape}")
print(f"Scaled features shape: {scaled_features.shape}")
print(f"Prepared real sequences shape for TimeGAN: {real_sequences.shape}")

# Ensure real_sequences is not empty before proceeding
if real_sequences.size == 0:
    raise ValueError("Real sequences are empty. Cannot train TimeGAN. Adjust SEQ_LEN or data range.")
```

### 3.2. Build and Train TimeGAN Model

Sarah now builds and trains the TimeGAN model. She will prioritize using `ydata-synthetic` for its optimized implementation. If it's not available, a simplified custom GAN for sequence generation is provided as a fallback (though `ydata-synthetic` is preferred for capturing advanced stylized facts).

```python
# Model Parameters for TimeGAN (from ydata-synthetic)
# These are crucial hyperparameters that affect training stability and generated data quality.
BATCH_SIZE = 128
TRAIN_STEPS = 5000 # Number of training iterations for TimeGAN
NOISE_DIM = 32 # Dimension of the random noise input to the generator
LAYERS_DIM = 128 # Hidden layer dimension for TimeGAN networks
DISCRIMINATOR_LOSSES = []
GENERATOR_LOSSES = []

if TIMEGAN_AVAILABLE:
    print("\nTraining TimeGAN using ydata-synthetic...")
    gan_args = ModelParameters(
        batch_size=BATCH_SIZE,
        lr=5e-4, # Learning rate
        noise_dim=NOISE_DIM,
        layers_dim=LAYERS_DIM,
        latent_dim=LATENT_DIM # Embedding dimension for TimeGAN
    )

    synth = TimeGAN(
        model_parameters=gan_args,
        n_seq=SEQ_LEN,
        n_features=N_FEATURES,
        gamma=1 # Supervised loss hyperparameter
    )
    
    # Train the TimeGAN model
    # ydata-synthetic handles the training loop internally and provides loss access
    synth.train(real_sequences, train_steps=TRAIN_STEPS)

    # Losses can be extracted from the synth object, e.g., synth.losses_g, synth.losses_d
    # For simplicity, we won't plot them here if using ydata-synthetic as they are often aggregated.
    # If a custom GAN was used, we'd plot the per-epoch losses.
    
    print("\nTimeGAN training completed using ydata-synthetic.")

else: # Fallback to custom simplified GAN if ydata-synthetic is not installed
    print("\nFalling back to custom simplified GAN implementation...")

    def build_generator(latent_dim, seq_len, n_features):
        model = Sequential([
            Dense(LAYERS_DIM, input_dim=latent_dim),
            LeakyReLU(0.2),
            BatchNormalization(),
            Dense(LAYERS_DIM * 2),
            LeakyReLU(0.2),
            BatchNormalization(),
            Dense(seq_len * n_features, activation='tanh'), # tanh outputs [-1, 1]
            Reshape((seq_len, n_features))
        ], name='generator')
        return model

    def build_discriminator(seq_len, n_features):
        model = Sequential([
            Flatten(input_shape=(seq_len, n_features)),
            Dense(LAYERS_DIM * 2),
            LeakyReLU(0.2),
            Dense(LAYERS_DIM),
            LeakyReLU(0.2),
            Dense(1, activation='sigmoid') # Binary classification (real/fake)
        ], name='discriminator')
        return model

    # Build and compile discriminator
    discriminator = build_discriminator(SEQ_LEN, N_FEATURES)
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

    # Build generator
    generator = build_generator(LATENT_DIM, SEQ_LEN, N_FEATURES)

    # Combined GAN model (generator trains via discriminator feedback)
    discriminator.trainable = False # Discriminator is not trained during GAN's generator training
    z_input = Input(shape=(LATENT_DIM,))
    fake_seq = generator(z_input)
    validity = discriminator(fake_seq)
    gan = Model(z_input, validity, name='GAN')
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                loss='binary_crossentropy')

    print(f"Generator params: {generator.count_params():,}")
    print(f"Discriminator params: {discriminator.count_params():,}")

    EPOCHS = 2000 # Number of training epochs for custom GAN

    d_losses, g_losses = [], []

    print(f"\nTraining custom GAN for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select a random batch of real sequences
        idx = np.random.randint(0, real_sequences.shape[0], BATCH_SIZE)
        real_batch = real_sequences[idx]

        # Generate a batch of fake sequences
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_batch = generator.predict(noise, verbose=0)

        # Train the discriminator
        # Label smoothing: real labels 0.9, fake labels 0.0 or 0.1
        d_loss_real = discriminator.train_on_batch(real_batch, np.ones((BATCH_SIZE, 1)) * 0.9)
        d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((BATCH_SIZE, 1)) * 0.1)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        # Generator wants discriminator to classify fake images as real
        g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: D loss={d_loss[0]:.4f}, G loss={g_loss:.4f}")
    
    DISCRIMINATOR_LOSSES = d_losses
    GENERATOR_LOSSES = g_losses
    print("\nCustom GAN training completed.")
```

### 3.3. Explanation of TimeGAN Training

Sarah understands that training GANs can be challenging, often exhibiting instability or mode collapse. The `ydata-synthetic` library simplifies this by encapsulating the complex TimeGAN architecture (embedding, generator, discriminator, recovery networks) and its specialized loss functions. The `train_steps` parameter dictates the duration of adversarial training, a critical factor for achieving high-quality synthetic data. If the custom GAN is used, the label smoothing (real labels `0.9`, fake `0.1` or `0.0`) is a common technique to stabilize training and prevent the discriminator from becoming too strong too quickly.

```python
if not TIMEGAN_AVAILABLE and GENERATOR_LOSSES: # Only plot if custom GAN was trained and losses exist
    plt.figure(figsize=(10, 5))
    plt.plot(GENERATOR_LOSSES, label='Generator Loss', alpha=0.8)
    plt.plot(DISCRIMINATOR_LOSSES, label='Discriminator Loss', alpha=0.8)
    plt.title('GAN Training Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('gan_loss_curves.png', dpi=150)
    plt.show()
```

---

## 4. Generating Synthetic Financial Paths

With the TimeGAN model trained, Sarah can now generate an ensemble of new synthetic financial time-series paths. This is the core output of the generative model, providing the augmented data needed for robust backtesting and stress testing. She aims to generate 1,000 synthetic sequences, each mirroring the length and features of her real historical sequences (20 days, OHLCV).

### 4.1. Generate Synthetic Sequences

Sarah uses the trained TimeGAN to generate a large number of synthetic sequences. These sequences are still in their scaled, multi-feature format.

```python
n_samples = 1000 # Number of synthetic sequences to generate

if TIMEGAN_AVAILABLE:
    synthetic_data_scaled = synth.sample(n_samples=n_samples)
else: # Use custom generator if ydata-synthetic not available
    noise_for_generation = np.random.normal(0, 1, (n_samples, LATENT_DIM))
    synthetic_data_scaled = generator.predict(noise_for_generation, verbose=0)

print(f"Generated synthetic data shape (scaled): {synthetic_data_scaled.shape}")
```

### 4.2. Inverse Transform and Extract Close Prices

The generated sequences are scaled and multi-featured. Sarah needs to inverse transform them back to their original scale and extract the 'close' price return column for comparison and strategy backtesting.

```python
# Create a dummy array for inverse transformation, ensuring correct shape and feature order
dummy_features = np.zeros((n_samples * SEQ_LEN, N_FEATURES))
dummy_features[:, :] = synthetic_data_scaled.reshape(n_samples * SEQ_LEN, N_FEATURES)

# Inverse transform the synthetic data
synthetic_data_original_scale = scaler.inverse_transform(dummy_features).reshape(n_samples, SEQ_LEN, N_FEATURES)

# Extract the 'close' return column (assuming 'close' is the 3rd feature, index 3)
synth_close_returns = synthetic_data_original_scale[:, :, 3]

# Convert synthetic returns to price paths, starting from S0=100
# For each synthetic path, create a cumulative product from returns and an initial price.
synth_prices = np.array([S0 * np.exp(np.cumsum(path_returns)) for path_returns in synth_close_returns])
synth_prices = np.column_stack([np.full(n_samples, S0), synth_prices]) # Add initial S0

print(f"Synthetic close returns shape: {synth_close_returns.shape}")
print(f"Synthetic prices shape: {synth_prices.shape}")

# Prepare real historical close returns for comparison
real_full_returns = features['close'].values # Use full historical returns for comparison with flattened synthetic returns
# Reshape real data into sequences similar to synthetic data structure for plotting
# Take enough real sequences to match number of synthetic paths for direct comparison in plots
num_real_paths_to_plot = min(n_samples, real_full_returns.shape[0] - SEQ_LEN + 1)
# Ensure prepare_sequences returns an array of shape (num_sequences, SEQ_LEN)
real_close_sequences_for_plot = prepare_sequences(real_full_returns.reshape(-1,1), SEQ_LEN).squeeze()[:num_real_paths_to_plot]

# Make sure real_close_sequences_for_plot is 2D for the loop
if real_close_sequences_for_plot.ndim == 1 and num_real_paths_to_plot > 0:
    real_close_sequences_for_plot = real_close_sequences_for_plot.reshape(num_real_paths_to_plot, -1)


real_prices_for_plot = np.array([S0 * np.exp(np.cumsum(path_returns)) for path_returns in real_close_sequences_for_plot])
real_prices_for_plot = np.column_stack([np.full(num_real_paths_to_plot, S0), real_prices_for_plot]) # Add initial S0

print(f"Real close sequences shape for plotting: {real_close_sequences_for_plot.shape}")
print(f"Real prices for plotting shape: {real_prices_for_plot.shape}")
```

### 4.3. Visualize Real vs. Synthetic Paths

Sarah visually inspects a subset of the generated TimeGAN price paths alongside real historical paths. This initial qualitative check helps confirm if the synthetic data "looks" realistic.

```python
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(1, 2, 1)
for i in range(min(20, num_real_paths_to_plot)): # Plot up to 20 real sample paths
    ax1.plot(real_prices_for_plot[i], alpha=0.4, color='blue')
ax1.set_title('Real 20-Day Return Paths (S&P 500 ETF)')
ax1.set_xlabel('Trading Day')
ax1.set_ylabel('Cumulative Price')
ax1.grid(True, linestyle=':', alpha=0.7)

ax2 = plt.subplot(1, 2, 2)
for i in range(20): # Plot 20 synthetic sample paths
    ax2.plot(synth_prices[i], alpha=0.4, color='red')
ax2.set_title('TimeGAN Synthetic 20-Day Return Paths')
ax2.set_xlabel('Trading Day')
ax2.set_ylabel('Cumulative Price')
ax2.grid(True, linestyle=':', alpha=0.7)

plt.suptitle('Real vs. Synthetic Financial Time Series', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('real_vs_synthetic_paths.png', dpi=150)
plt.show()
```

### 4.4. Explanation of Synthetic Path Generation

The generation process transforms random noise into structured financial time series. The visual comparison is a quick sanity check. While the synthetic paths should mimic the overall pattern of real paths (e.g., general trends, volatility levels), they won't be identical, which is the point: generating *new, plausible* scenarios. This visual validation is a precursor to more rigorous statistical assessment.

---

## 5. Statistical Quality Assessment of Synthetic Data

Visual inspection is not enough. Sarah needs to quantitatively assess how well the synthetic data captures the key statistical properties and "stylized facts" of real financial returns, especially comparing TimeGAN against the GBM baseline. This rigorous evaluation directly addresses the limitations of traditional models and verifies the value of the generative approach.

### 5.1. Compare Descriptive Statistics

Sarah computes the mean, standard deviation, skewness, and kurtosis for real, TimeGAN-generated, and GBM-simulated close returns. She's looking for TimeGAN to better match the fat tails (high kurtosis, $Kurtosis > 3$) and negative skewness often observed in real financial data, which GBM typically misses.

```python
# Flatten the return arrays for descriptive statistics and KS test
real_flat = real_full_returns # Use the full historical close returns for comparison
synth_flat = synth_close_returns.flatten()
gbm_flat = gbm_returns.flatten()[:len(real_flat)] # Match length for fair comparison

print("Descriptive Statistics Comparison:")
print(f"{'Metric':<20s} {'Real':>10s} {'Synthetic (TimeGAN)':>20s} {'GBM':>10s}")
print("-" * 62)

metrics_to_compare = [
    ('Mean', np.mean),
    ('Std', np.std),
    ('Skewness', lambda x: float(describe(x).skewness)),
    ('Kurtosis', lambda x: float(describe(x).kurtosis))
]

for name, func in metrics_to_compare:
    r_val = func(real_flat)
    s_val = func(synth_flat)
    g_val = func(gbm_flat)
    print(f"{name:<20s} {r_val:>10.5f} {s_val:>20.5f} {g_val:>10.5f}")
```

### 5.2. Kolmogorov-Smirnov (KS) Test for Distributional Similarity

The **Kolmogorov-Smirnov (KS) test** quantifies the maximum difference between the cumulative distribution functions (CDFs) of two samples. A lower KS statistic and a higher p-value (typically $p > 0.05$) indicate that we cannot reject the null hypothesis that the two samples are drawn from the same underlying distribution. Sarah uses this to compare the distributional similarity of synthetic returns to real returns.

The KS statistic $D_{KS}$ is defined as:
$$ D_{KS} = \sup_x |F_{\text{real}}(x) - F_{\text{synth}}(x)| $$
where $F_{\text{real}}(x)$ and $F_{\text{synth}}(x)$ are the empirical cumulative distribution functions of the real and synthetic data, respectively.

```python
# Perform KS test
ks_gan, p_gan = ks_2samp(real_flat, synth_flat)
ks_gbm, p_gbm = ks_2samp(real_flat, gbm_flat)

print(f"\nKolmogorov-Smirnov Test Results:")
print(f"TimeGAN vs Real: KS Stat={ks_gan:.4f} (p={p_gan:.4f})")
print(f"GBM vs Real:     KS Stat={ks_gbm:.4f} (p={p_gbm:.4f})")
print("(Lower KS statistic and higher p-value indicate more similar distributions)")
```

### 5.3. Compare Return Distributions (Histograms)

Sarah generates histograms to visually overlay the return distributions for real, TimeGAN, and GBM data. This visualization complements the descriptive statistics and KS test, allowing her to visually confirm the capture of fat tails.

```python
plt.figure(figsize=(12, 6))
sns.histplot(real_flat, bins=50, color='blue', label='Real Returns', kde=True, stat='density', alpha=0.6)
sns.histplot(synth_flat, bins=50, color='red', label='TimeGAN Synthetic Returns', kde=True, stat='density', alpha=0.6)
sns.histplot(gbm_flat, bins=50, color='green', label='GBM Simulated Returns', kde=True, stat='density', alpha=0.6)
plt.title('Distribution of Daily Close Returns: Real vs. Synthetic vs. GBM')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('return_distributions_overlay.png', dpi=150)
plt.show()
```

### 5.4. Autocorrelation Function (ACF) Comparison

Volatility clustering, where periods of high volatility are followed by high volatility, is a critical stylized fact. This often manifests as significant autocorrelation in the absolute returns. GBM, by construction, has no autocorrelation. Sarah computes and compares the Autocorrelation Function (ACF) for real, TimeGAN, and GBM returns (and absolute returns) up to a specified lag.

```python
def compute_acf(x, nlags=20):
    """
    Computes the Autocorrelation Function (ACF) for a given series.
    """
    result = []
    x_demean = x - np.mean(x)
    for lag in range(nlags + 1):
        if lag == 0:
            result.append(1.0)
        else:
            # Ensure slices are of equal length for np.corrcoef
            if len(x_demean[:-lag]) == len(x_demean[lag:]):
                corr = np.corrcoef(x_demean[:-lag], x_demean[lag:])[0, 1]
            else:
                corr = np.nan # Or handle error appropriately
            result.append(corr)
    return [r for r in result if not np.isnan(r)] # Filter out NaNs if any

nlags = 20
acf_real = compute_acf(real_flat, nlags)
acf_synth = compute_acf(synth_flat, nlags)
acf_gbm = compute_acf(gbm_flat, nlags)

# ACF of absolute returns for volatility clustering
acf_abs_real = compute_acf(np.abs(real_flat), nlags)
acf_abs_synth = compute_acf(np.abs(synth_flat), nlags)
acf_abs_gbm = compute_acf(np.abs(gbm_flat), nlags)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(range(len(acf_real)), acf_real, label='Real Returns', marker='o', linestyle='--')
plt.plot(range(len(acf_synth)), acf_synth, label='TimeGAN Synthetic Returns', marker='x', linestyle='-.')
plt.plot(range(len(acf_gbm)), acf_gbm, label='GBM Simulated Returns', marker='^', linestyle=':')
plt.title('Autocorrelation Function (ACF) of Returns')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(acf_abs_real)), acf_abs_real, label='Real Absolute Returns', marker='o', linestyle='--')
plt.plot(range(len(acf_abs_synth)), acf_abs_synth, label='TimeGAN Synthetic Absolute Returns', marker='x', linestyle='-.')
plt.plot(range(len(acf_abs_gbm)), acf_abs_gbm, label='GBM Simulated Absolute Returns', marker='^', linestyle=':')
plt.title('Autocorrelation Function (ACF) of Absolute Returns (Volatility Clustering)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('acf_comparison.png', dpi=150)
plt.show()
```

### 5.5. t-SNE Visualization for Manifold Overlap

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a dimensionality reduction technique particularly effective for visualizing high-dimensional data in 2D or 3D. Sarah uses t-SNE to project the real and synthetic time sequences into a 2D space. Significant overlap between the real and synthetic data clusters indicates that the TimeGAN has learned to generate data that resides on the same manifold as the real data, suggesting high fidelity and realism. Separation, conversely, might indicate mode collapse or a distributional mismatch.

```python
# Sample equal numbers of real and synthetic sequences for t-SNE visualization
n_viz = min(500, real_sequences.shape[0], synthetic_data_scaled.shape[0])

# Reshape sequences for t-SNE (flatten each sequence into a single row)
real_sequences_flat_viz = real_sequences[:n_viz].reshape(n_viz, -1)
synthetic_sequences_flat_viz = synthetic_data_scaled[:n_viz].reshape(n_viz, -1)

combined_sequences_flat = np.vstack([real_sequences_flat_viz, synthetic_sequences_flat_viz])

labels_viz = ['Real'] * n_viz + ['Synthetic'] * n_viz

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
embedded_sequences = tsne.fit_transform(combined_sequences_flat)

plt.figure(figsize=(10, 8))
plt.scatter(embedded_sequences[:n_viz, 0], embedded_sequences[:n_viz, 1],
            c='blue', alpha=0.5, s=20, label='Real')
plt.scatter(embedded_sequences[n_viz:, 0], embedded_sequences[n_viz:, 1],
            c='red', alpha=0.5, s=20, label='Synthetic')
plt.title('t-SNE: Real vs. Synthetic Return Sequences Manifold Overlap')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('tsne_real_vs_synthetic.png', dpi=150)
plt.show()
```

### 5.6. Explanation of Statistical Quality Assessment

This comprehensive assessment confirms TimeGAN's ability to capture financial stylized facts beyond what GBM can. Sarah can now confidently present her findings:
*   **Descriptive Statistics & Histograms:** TimeGAN should show kurtosis values closer to real data (indicating fat tails) and potentially better skewness than GBM, which defaults to normal-like distributions.
*   **KS Test:** A lower KS statistic and higher p-value for TimeGAN compared to GBM suggest that TimeGAN's generated distribution is statistically more similar to the real data.
*   **ACF Plots:** TimeGAN should capture autocorrelation in returns (especially in short lags for real data) and, crucially, in absolute returns (indicating volatility clustering), which GBM will fail to replicate.
*   **t-SNE:** Significant overlap indicates that the synthetic data points occupy the same underlying data space or "manifold" as the real data, implying high-fidelity generation.

This quantitative evidence is critical for justifying the use of TimeGAN-generated data for subsequent analysis steps.

---

## 6. Strategy Backtesting on Synthetic Data

The ultimate goal for Sarah is to use the generated synthetic paths to rigorously backtest investment strategies. Instead of a single point estimate of performance from one historical path, she can now obtain a *distribution* of performance metrics (e.g., Sharpe ratios) across thousands of plausible market scenarios. This provides a more robust understanding of a strategy's true performance and its resilience to varying market conditions, directly addressing concerns about backtest overfitting.

### 6.1. Implement a Simple Momentum Strategy

Sarah defines a simple rule-based momentum strategy: go long if the sum of returns over a `lookback` period is positive.

```python
def momentum_strategy(returns, lookback=10):
    """
    Implements a simple momentum strategy.
    Goes long if the sum of returns over the lookback period is positive, otherwise flat.
    
    Args:
        returns (np.array): A 1D array of daily returns.
        lookback (int): The number of days to look back for momentum calculation.
        
    Returns:
        np.array: Daily strategy returns.
    """
    signals = np.zeros_like(returns)
    # Ensure lookback is less than the length of returns
    if len(returns) <= lookback:
        return signals # Cannot apply strategy if not enough history

    for t in range(lookback, len(returns)):
        if np.sum(returns[t-lookback:t]) > 0:
            signals[t] = 1.0 # Go long
        # else: signals[t] = 0.0 # Stay flat (cash)
            
    strategy_returns = signals * returns
    return strategy_returns

def calculate_sharpe_ratio(returns, annual_risk_free_rate=0.02, trading_days_per_year=252):
    """
    Calculates the annualized Sharpe Ratio.
    """
    if np.std(returns) == 0:
        return 0.0 # Avoid division by zero for flat returns
    
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/trading_days_per_year) - 1
    # Ensure returns array is not empty before calculating mean
    if returns.size == 0:
        return 0.0

    excess_returns = returns - daily_risk_free_rate
    
    annualized_mean_excess_return = np.mean(excess_returns) * trading_days_per_year
    annualized_std_dev = np.std(returns) * np.sqrt(trading_days_per_year)
    
    sharpe_ratio = annualized_mean_excess_return / annualized_std_dev
    return sharpe_ratio

# Define lookback for the strategy
STRATEGY_LOOKBACK = 10
```

### 6.2. Backtest on Real Historical Data

First, Sarah backtests the momentum strategy on the entire historical 'close' returns to get a baseline Sharpe ratio.

```python
# Use the full historical close returns for real backtest
real_full_returns = features['close'].values 
real_strat_returns = momentum_strategy(real_full_returns, lookback=STRATEGY_LOOKBACK)
real_sharpe_ratio = calculate_sharpe_ratio(real_strat_returns)

print(f"Strategy Sharpe Ratio on real historical data: {real_sharpe_ratio:.3f}")
```

### 6.3. Backtest on Synthetic Data Ensemble

Now, Sarah applies the same momentum strategy to each of the 1,000 synthetic return paths generated by TimeGAN. This yields an ensemble of Sharpe ratios, which can then be analyzed as a distribution.

```python
synth_sharpe_ratios = []

# Iterate through each synthetic return path
for i in range(synth_close_returns.shape[0]):
    path_returns = synth_close_returns[i]
    strat_returns_synth = momentum_strategy(path_returns, lookback=STRATEGY_LOOKBACK)
    
    # Calculate Sharpe ratio for the synthetic path
    sharpe = calculate_sharpe_ratio(strat_returns_synth)
    synth_sharpe_ratios.append(sharpe)

synth_sharpe_ratios = np.array(synth_sharpe_ratios)

print(f"\nStrategy Sharpe Ratio on synthetic data (Mean): {np.mean(synth_sharpe_ratios):.3f}")
print(f"Strategy Sharpe Ratio on synthetic data (5th Percentile): {np.percentile(synth_sharpe_ratios, 5):.3f}")
print(f"Strategy Sharpe Ratio on synthetic data (95th Percentile): {np.percentile(synth_sharpe_ratios, 95):.3f}")
```

### 6.4. Visualize Distribution of Synthetic Sharpe Ratios

Sarah visualizes the distribution of Sharpe ratios from the synthetic backtests using a histogram. She also marks the real historical Sharpe ratio on this histogram for comparison. This visualization is the key deliverable for assessing strategy robustness.

```python
plt.figure(figsize=(10, 6))
plt.hist(synth_sharpe_ratios, bins=50, alpha=0.7, edgecolor='black', color='coral', label='Synthetic Sharpe Ratios')
plt.axvline(x=real_sharpe_ratio, color='blue', linestyle='--', linewidth=2, label=f'Real Sharpe: {real_sharpe_ratio:.2f}')
plt.title('Distribution of Strategy Performance Across 1000 Synthetic Paths')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('synthetic_backtest_distribution.png', dpi=150)
plt.show()
```

### 6.5. Explanation of Synthetic Backtest Distribution

This distribution of Sharpe ratios is the financial centerpiece for Sarah. Instead of a single point estimate, she now has a confidence interval for the strategy's performance.
*   If the real historical Sharpe ratio falls within the central part of the synthetic distribution, it suggests the strategy is robust and its historical performance is plausible across a range of scenarios.
*   If the real Sharpe ratio is an extreme outlier (e.g., above the 95th percentile), it could indicate that the strategy's historical performance was fortuitous or that the backtest is overfitted to the single historical path.
This approach significantly enhances Sarah's ability to evaluate strategy robustness, informing whether to deploy or refine the strategy. It moves beyond "what happened" to "what could happen."

---

## 7. Addressing GAN Limitations: Mode Collapse and Training Stability

While powerful, GANs, including TimeGAN, come with inherent challenges that Sarah, as a quantitative strategist, must be aware of. Understanding these limitations is crucial for correctly interpreting results and troubleshooting.

### 7.1. Discussion: Mode Collapse, Training Instability, and Hyperparameter Sensitivity

*   **Mode Collapse:** This is a primary GAN failure mode. It occurs when the generator learns to produce only a limited variety of samples, essentially "collapsing" to a few modes of the real data distribution.
    *   **Symptoms:** Low diversity in generated samples; all generated paths looking too similar; t-SNE plot showing synthetic data clustered tightly in a small region, failing to cover the entire real data manifold. Unrealistically low standard deviation across synthetic paths compared to real data.
    *   **Detection:** Computing the inter-path standard deviation of synthetic data and comparing it to real historical data (e.g., `np.std(synth_close_returns.std(axis=1))` vs `np.std(real_close_sequences_for_plot.std(axis=1))`). If the synthetic value is significantly lower, it's a red flag.
    *   **Mitigation:** Techniques like Wasserstein GANs (WGANs) with gradient penalty, feature matching, or training with a greater diversity of inputs can help. TimeGAN's supervised loss also inherently encourages temporal diversity.

*   **Training Instability:** GANs are notoriously hard to train. The adversarial min-max game between the generator and discriminator can be unstable, leading to oscillating or non-converging loss curves (as seen in Section 3.3 for custom GANs).
    *   **Symptoms:** Generator and discriminator losses fluctuating wildly or not converging to a stable equilibrium. Generated sample quality varying greatly across epochs.
    *   **Mitigation:** Careful hyperparameter tuning (learning rates, batch size), using WGANs, or applying label smoothing can improve stability.

*   **Hyperparameter Sensitivity:** The quality of GAN-generated data is highly sensitive to hyperparameters like learning rate, batch size, network architecture, and loss weights ($\gamma$ in TimeGAN). A configuration that works for one dataset may not work for another.
    *   **Symptoms:** Poor quality synthetic data, even after extensive training, due to suboptimal hyperparameter choices.
    *   **Mitigation:** Extensive hyperparameter search, cross-validation, and potentially transfer learning from pre-trained models.

*   **Non-stationarity:** A GAN trained on historical data learns the market dynamics of that specific period. Using it to generate future scenarios implicitly assumes stationarity of these dynamics, which is often violated in financial markets.
*   **Validation Paradox:** If we could perfectly evaluate synthetic data quality, we wouldn't need synthetic data—we would already understand the true underlying distribution. This highlights the inherent challenge: we are using synthetic data because we don't fully understand the real data's generative process, yet we need to validate the synthetic data against that same unknown process.

Sarah understands that while TimeGAN provides a powerful tool, it requires careful implementation, monitoring, and a critical understanding of its potential pitfalls to ensure the generated data is truly valuable for financial decision-making.
