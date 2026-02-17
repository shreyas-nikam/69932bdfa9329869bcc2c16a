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
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input)
from tensorflow.keras.optimizers import Adam
import warnings
import os

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

# --- Helper Functions ---

def load_and_preprocess_data(ticker: str = 'SPY', start_date: str = '2010-01-01', end_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Loads historical OHLCV data, calculates percentage changes for prices,
    and log-transformed differences for volume.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame with preprocessed features (pct_change for OHLC, log_diff for Volume).

    Raises:
        ValueError: If no data is downloaded or insufficient data for processing.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker and date range.")

    if len(data) < 2:
        raise ValueError(f"Insufficient data downloaded for {ticker}. Expected at least 2 rows, but got {len(data)}. Adjust date range or ticker.")

    features_df = data[['Open', 'High', 'Low', 'Close']].pct_change()
    features_df['volume'] = np.log1p(data['Volume']).diff()
    features_df = features_df.dropna()

    print(f"\nTraining data shape: {features_df.shape[0]} days x {features_df.shape[1]} features")
    close_returns = features_df['Close']
    mean_val = float(close_returns.mean())
    std_val = float(close_returns.std())
    skew_val = float(close_returns.skew())
    kurt_val = float(close_returns.kurtosis())
    print(f"Close return stats: mean={mean_val:.5f}, std={std_val:.4f}, "
          f"skew={skew_val:.2f}, kurt={kurt_val:.2f}")

    return features_df

def simulate_gbm(mu_annual: float, sigma_annual: float, S0: float, n_days: int, n_paths: int, dt_annual: float = 1/252) -> tuple[np.ndarray, np.ndarray]:
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
    daily_returns_factor = np.exp((mu_annual - 0.5 * sigma_annual**2) * dt_annual + sigma_annual * np.sqrt(dt_annual) * np.random.standard_normal((n_paths, n_days)))
    prices = np.full((n_paths, n_days + 1), S0, dtype=float)
    for t in range(n_days):
        prices[:, t+1] = prices[:, t] * daily_returns_factor[:, t]
    returns = np.log(prices[:, 1:] / prices[:, :-1])
    return returns, prices

def prepare_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Creates sliding-window sequences for GAN training.

    Args:
        data (np.array): Input data array (e.g., scaled features).
        seq_len (int): Length of each sequence.

    Returns:
        np.array: Array of sequences.
    """
    sequences = []
    if len(data) < seq_len:
        print(f"Warning: Not enough data points ({len(data)}) to create sequences of desired length ({seq_len}).")
        return np.array([])
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

def build_generator(latent_dim: int, seq_len: int, n_features: int, layers_dim: int) -> tf.keras.Model:
    """
    Builds the generator model for a custom GAN.

    Args:
        latent_dim (int): Dimension of the noise vector.
        seq_len (int): Length of output sequences.
        n_features (int): Number of features in output sequences.
        layers_dim (int): Dimension of hidden layers.

    Returns:
        tf.keras.Model: Compiled generator model.
    """
    model = Sequential([
        Dense(layers_dim, input_dim=latent_dim),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(layers_dim * 2),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(seq_len * n_features, activation='tanh'),
        Reshape((seq_len, n_features))
    ], name='generator')
    return model

def build_discriminator(seq_len: int, n_features: int, layers_dim: int) -> tf.keras.Model:
    """
    Builds the discriminator model for a custom GAN.

    Args:
        seq_len (int): Length of input sequences.
        n_features (int): Number of features in input sequences.
        layers_dim (int): Dimension of hidden layers.

    Returns:
        tf.keras.Model: Compiled discriminator model.
    """
    model = Sequential([
        Flatten(input_shape=(seq_len, n_features)),
        Dense(layers_dim * 2),
        LeakyReLU(0.2),
        Dense(layers_dim),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ], name='discriminator')
    return model

def compute_acf(x: np.ndarray, nlags: int = 20) -> list:
    """
    Computes the Autocorrelation Function (ACF) for a given series.

    Args:
        x (np.array): Input time series.
        nlags (int): Number of lags to compute ACF for.

    Returns:
        list: List of autocorrelation values.
    """
    result = []
    x_demean = x - np.mean(x)
    for lag in range(nlags + 1):
        if lag == 0:
            result.append(1.0)
        else:
            if len(x_demean[:-lag]) > 0 and len(x_demean[lag:]) > 0:
                min_len = min(len(x_demean[:-lag]), len(x_demean[lag:]))
                if min_len > 0:
                    # Check for zero variance to avoid NaNs from corrcoef
                    if np.std(x_demean[:-lag][:min_len]) == 0 or np.std(x_demean[lag:][:min_len]) == 0:
                        corr = 0.0 # If no variance, correlation is undefined, treat as 0
                    else:
                        corr = np.corrcoef(x_demean[:-lag][:min_len], x_demean[lag:][:min_len])[0, 1]
                else:
                    corr = np.nan
            else:
                corr = np.nan
            result.append(corr)
    return [r for r in result if not np.isnan(r)]

def momentum_strategy(returns: np.ndarray, lookback: int = 10) -> np.ndarray:
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
    if len(returns) <= lookback:
        return signals

    for t in range(lookback, len(returns)):
        if np.sum(returns[t-lookback:t]) > 0:
            signals[t] = 1.0

    strategy_returns = signals * returns
    return strategy_returns

def calculate_sharpe_ratio(returns: np.ndarray, annual_risk_free_rate: float = 0.02, trading_days_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe Ratio.

    Args:
        returns (np.array): Array of daily returns.
        annual_risk_free_rate (float): Annual risk-free rate.
        trading_days_per_year (int): Number of trading days in a year.

    Returns:
        float: Annualized Sharpe ratio.
    """
    if returns.size == 0 or np.std(returns) == 0:
        return 0.0

    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/trading_days_per_year) - 1
    excess_returns = returns - daily_risk_free_rate

    annualized_mean_excess_return = np.mean(excess_returns) * trading_days_per_year
    annualized_std_dev = np.std(returns) * np.sqrt(trading_days_per_year)

    sharpe_ratio = annualized_mean_excess_return / annualized_std_dev
    return sharpe_ratio

# --- Main Functions for the Workflow ---

def run_gbm_simulation_and_plot(
    features_df: pd.DataFrame,
    s0: float = 100,
    n_days_sim: int = 252,
    n_paths_sim: int = 1000,
    output_path: str = 'gbm_sample_paths.png'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calibrates GBM parameters from historical data, simulates price paths, and plots them.

    Args:
        features_df (pd.DataFrame): DataFrame containing preprocessed historical features.
        s0 (float): Initial price for GBM simulation.
        n_days_sim (int): Number of days to simulate for each path.
        n_paths_sim (int): Number of independent price paths to generate.
        output_path (str): File path to save the GBM plot.

    Returns:
        tuple[np.ndarray, np.ndarray]: (gbm_returns, gbm_prices)
    """
    historical_close_returns = features_df['Close'].values
    mu_annual = np.mean(historical_close_returns) * 252
    sigma_annual = np.std(historical_close_returns) * np.sqrt(252)

    print(f"\nCalibrated GBM: Annual Mean (mu) = {mu_annual:.4f}, Annual Std (sigma) = {sigma_annual:.4f}")
    gbm_returns, gbm_prices = simulate_gbm(mu_annual, sigma_annual, s0, n_days_sim, n_paths_sim)

    print(f"GBM simulated paths shape (returns): {gbm_returns.shape}")
    print(f"GBM simulated paths shape (prices): {gbm_prices.shape}")

    plt.figure(figsize=(12, 6))
    for i in range(min(20, n_paths_sim)):
        plt.plot(gbm_prices[i], alpha=0.5, color='blue')
    plt.title(f'Monte Carlo GBM: {min(20, n_paths_sim)} Sample Price Paths (S&P 500 ETF)')
    plt.xlabel('Trading Day')
    plt.ylabel('Price')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"GBM sample paths plot saved to {output_path}")

    return gbm_returns, gbm_prices

def prepare_gan_training_data(
    features_df: pd.DataFrame,
    seq_len: int
) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Scales features and prepares them into sequences for GAN training.

    Args:
        features_df (pd.DataFrame): Preprocessed historical features.
        seq_len (int): Length of each sequence.

    Returns:
        tuple[np.ndarray, MinMaxScaler]: (scaled_sequences, fitted_scaler)

    Raises:
        ValueError: If no real sequences can be formed.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_features = scaler.fit_transform(features_df.values)
    real_sequences = prepare_sequences(scaled_features, seq_len)

    print(f"Original features shape: {features_df.shape}")
    print(f"Scaled features shape: {scaled_features.shape}")
    print(f"Prepared real sequences shape for GAN: {real_sequences.shape}")

    if real_sequences.size == 0:
        raise ValueError("Real sequences are empty. Cannot train GAN. Adjust SEQ_LEN or data range.")

    return real_sequences, scaler

def train_gan_model(
    real_sequences: np.ndarray,
    seq_len: int,
    n_features: int,
    batch_size: int = 128,
    timegan_train_steps: int = 5000,
    custom_gan_epochs: int = 200,
    noise_dim: int = 32,
    layers_dim: int = 128,
    latent_dim: int = 32,
    timegan_available: bool = False
) -> tuple[tf.keras.Model | object, list, list]:
    """
    Trains a GAN model (either ydata-synthetic's TimeGAN or a custom simplified GAN).

    Args:
        real_sequences (np.ndarray): Scaled historical sequences.
        seq_len (int): Length of sequences.
        n_features (int): Number of features per sequence.
        batch_size (int): Training batch size.
        timegan_train_steps (int): Number of training steps for ydata-synthetic TimeGAN.
        custom_gan_epochs (int): Number of epochs for custom GAN.
        noise_dim (int): Dimension of noise vector for generator.
        layers_dim (int): Dimension of hidden layers in GAN networks.
        latent_dim (int): Latent dimension for generator.
        timegan_available (bool): Flag indicating if ydata-synthetic TimeGAN is available.

    Returns:
        tuple[tf.keras.Model | object, list, list]:
            - Trained GAN model (TimeGAN object or custom Keras generator).
            - List of discriminator losses (empty if TimeGAN).
            - List of generator losses (empty if TimeGAN).
    """
    d_losses, g_losses = [], []
    trained_model = None

    if timegan_available:
        print("\nTraining TimeGAN using ydata-synthetic...")
        gan_args = ModelParameters(
            batch_size=batch_size,
            lr=5e-4,
            noise_dim=noise_dim,
            layers_dim=layers_dim,
            latent_dim=latent_dim
        )
        synth = TimeGAN(
            model_parameters=gan_args,
            n_seq=seq_len,
            n_features=n_features,
            gamma=1
        )
        synth.train(real_sequences, train_steps=timegan_train_steps)
        trained_model = synth
        print("\nTimeGAN training completed using ydata-synthetic.")
    else:
        print("\nFalling back to custom simplified GAN implementation...")
        generator = build_generator(latent_dim, seq_len, n_features, layers_dim)
        discriminator = build_discriminator(seq_len, n_features, layers_dim)

        discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

        discriminator.trainable = False
        z_input = Input(shape=(latent_dim,))
        fake_seq = generator(z_input)
        validity = discriminator(fake_seq)
        gan = Model(z_input, validity, name='GAN')
        gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                    loss='binary_crossentropy')

        print(f"Generator params: {generator.count_params():,}")
        print(f"Discriminator params: {discriminator.count_params():,}")

        print(f"\nTraining custom GAN for {custom_gan_epochs} epochs...")
        for epoch in range(custom_gan_epochs):
            idx = np.random.randint(0, real_sequences.shape[0], batch_size)
            real_batch = real_sequences[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_batch = generator.predict(noise, verbose=0)

            d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)) * 0.9)
            d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)) * 0.1)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if epoch % (custom_gan_epochs // 10 if custom_gan_epochs >= 10 else 1) == 0:
                print(f"Epoch {epoch}/{custom_gan_epochs}: D loss={d_loss[0]:.4f}, G loss={g_loss:.4f}")
        trained_model = generator
        print("\nCustom GAN training completed.")

    return trained_model, d_losses, g_losses

def plot_gan_losses(
    generator_losses: list,
    discriminator_losses: list,
    output_path: str = 'gan_loss_curves.png'
):
    """
    Plots the generator and discriminator losses during GAN training.

    Args:
        generator_losses (list): List of generator loss values.
        discriminator_losses (list): List of discriminator loss values.
        output_path (str): File path to save the loss plot.
    """
    if generator_losses and discriminator_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(generator_losses, label='Generator Loss', alpha=0.8)
        plt.plot(discriminator_losses, label='Discriminator Loss', alpha=0.8)
        plt.title('GAN Training Losses over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"GAN loss curves plot saved to {output_path}")
    else:
        print("No GAN loss data available for plotting (ydata-synthetic TimeGAN does not expose per-epoch losses this way, or training was skipped).")

def generate_synthetic_data(
    trained_model: tf.keras.Model | object,
    n_samples: int,
    seq_len: int,
    n_features: int,
    latent_dim: int,
    scaler: MinMaxScaler,
    timegan_available: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic data using the trained GAN model and inverse transforms it.

    Args:
        trained_model (tf.keras.Model | object): The trained GAN generator model or TimeGAN object.
        n_samples (int): Number of synthetic sequences to generate.
        seq_len (int): Length of each sequence.
        n_features (int): Number of features per sequence.
        latent_dim (int): Latent dimension used for the generator's noise input.
        scaler (MinMaxScaler): The fitted scaler used to transform real data.
        timegan_available (bool): Flag indicating if ydata-synthetic TimeGAN was used.

    Returns:
        tuple[np.ndarray, np.ndarray]: (synthetic_data_original_scale, synthetic_data_scaled)
    """
    if timegan_available:
        synthetic_data_scaled = trained_model.sample(n_samples=n_samples)
    else:
        noise_for_generation = np.random.normal(0, 1, (n_samples, latent_dim))
        synthetic_data_scaled = trained_model.predict(noise_for_generation, verbose=0)

    print(f"Generated synthetic data shape (scaled): {synthetic_data_scaled.shape}")

    dummy_features = np.zeros((n_samples * seq_len, n_features))
    dummy_features[:, :] = synthetic_data_scaled.reshape(n_samples * seq_len, n_features)
    synthetic_data_original_scale = scaler.inverse_transform(dummy_features).reshape(n_samples, seq_len, n_features)

    return synthetic_data_original_scale, synthetic_data_scaled

def plot_real_vs_synthetic_paths(
    features_df: pd.DataFrame,
    synthetic_data_original_scale: np.ndarray,
    s0: float = 100,
    seq_len: int = 20,
    n_samples_to_plot: int = 20,
    output_path: str = 'real_vs_synthetic_paths.png'
) -> np.ndarray:
    """
    Generates price paths from real and synthetic returns and plots a comparison.

    Args:
        features_df (pd.DataFrame): DataFrame with real historical features.
        synthetic_data_original_scale (np.ndarray): Generated synthetic data in original scale.
        s0 (float): Starting price for all paths.
        seq_len (int): Length of each sequence.
        n_samples_to_plot (int): Number of sample paths to plot.
        output_path (str): File path to save the plot.

    Returns:
        np.ndarray: Synthetic close returns (flattened).
    """
    # Assuming 'Close' is the 3rd feature (index 3) after Open, High, Low
    synth_close_returns = synthetic_data_original_scale[:, :, 3]

    # Convert synthetic returns to price paths
    synth_prices = np.array([s0 * np.exp(np.cumsum(path_returns)) for path_returns in synth_close_returns])
    synth_prices = np.column_stack([np.full(synth_close_returns.shape[0], s0), synth_prices])

    real_full_returns = features_df['Close'].values
    num_real_paths_to_plot = min(n_samples_to_plot, real_full_returns.shape[0] - seq_len + 1)

    if num_real_paths_to_plot <= 0:
        print("Warning: Not enough real data to create sequences for plotting. Skipping real path plot.")
        real_prices_for_plot = np.array([])
        real_close_sequences_for_plot = np.array([])
    else:
        real_close_sequences_for_plot = prepare_sequences(real_full_returns.reshape(-1, 1), seq_len).squeeze()
        if real_close_sequences_for_plot.ndim == 1 and num_real_paths_to_plot > 0:
            real_close_sequences_for_plot = real_close_sequences_for_plot.reshape(-1, seq_len)
        real_close_sequences_for_plot = real_close_sequences_for_plot[:num_real_paths_to_plot]

        real_prices_for_plot = np.array([s0 * np.exp(np.cumsum(path_returns)) for path_returns in real_close_sequences_for_plot])
        real_prices_for_plot = np.column_stack([np.full(num_real_paths_to_plot, s0), real_prices_for_plot])
        print(f"Real prices for plotting shape: {real_prices_for_plot.shape}")

    print(f"Synthetic close returns shape: {synth_close_returns.shape}")
    print(f"Synthetic prices shape: {synth_prices.shape}")
    print(f"Real close sequences shape for plotting: {real_close_sequences_for_plot.shape}")


    plt.figure(figsize=(16, 6))

    ax1 = plt.subplot(1, 2, 1)
    if real_prices_for_plot.size > 0:
        for i in range(min(n_samples_to_plot, num_real_paths_to_plot)):
            ax1.plot(real_prices_for_plot[i], alpha=0.4, color='blue')
    ax1.set_title(f'Real {seq_len}-Day Return Paths (S&P 500 ETF)')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Cumulative Price')
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2 = plt.subplot(1, 2, 2)
    for i in range(min(n_samples_to_plot, synth_prices.shape[0])):
        ax2.plot(synth_prices[i], alpha=0.4, color='red')
    ax2.set_title(f'TimeGAN Synthetic {seq_len}-Day Return Paths')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Cumulative Price')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.suptitle('Real vs. Synthetic Financial Time Series', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Real vs. Synthetic paths plot saved to {output_path}")

    return synth_close_returns

def perform_statistical_comparison(
    real_features_df: pd.DataFrame,
    synthetic_close_returns: np.ndarray,
    gbm_returns: np.ndarray,
    output_path: str = 'return_distributions_overlay.png'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compares real, synthetic, and GBM returns using descriptive statistics, KS test, and histograms.

    Args:
        real_features_df (pd.DataFrame): DataFrame with real historical features.
        synthetic_close_returns (np.ndarray): Synthetic close returns.
        gbm_returns (np.ndarray): GBM simulated returns.
        output_path (str): File path to save the histogram plot.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (real_flat, synth_flat, gbm_flat)
    """
    real_flat = real_features_df['Close'].values.flatten()
    synth_flat = synthetic_close_returns.flatten()
    gbm_flat = gbm_returns.flatten()[:len(real_flat)]

    print("\nDescriptive Statistics Comparison:")
    print(f"{'Metric':<20s} {'Real':>10s} {'Synthetic (TimeGAN)':>20s} {'GBM':>10s}")
    print("-" * 62)

    metrics_to_compare = [
        ('Mean', np.mean),
        ('Std', np.std),
        ('Skewness', lambda x: float(describe(x).skewness) if x.size > 1 else np.nan), # describe needs at least 2 samples
        ('Kurtosis', lambda x: float(describe(x).kurtosis) if x.size > 1 else np.nan)
    ]

    for name, func in metrics_to_compare:
        r_val = func(real_flat)
        s_val = func(synth_flat)
        g_val = func(gbm_flat)
        print(f"{name:<20s} {r_val:>10.5f} {s_val:>20.5f} {g_val:>10.5f}")

    ks_gan, p_gan = ks_2samp(real_flat, synth_flat)
    ks_gbm, p_gbm = ks_2samp(real_flat, gbm_flat)

    print(f"\nKolmogorov-Smirnov Test Results:")
    print(f"TimeGAN vs Real: KS Stat={ks_gan.item():.4f} (p={p_gan.item():.4f})")
    print(f"GBM vs Real:     KS Stat={ks_gbm.item():.4f} (p={p_gbm.item():.4f})")
    print("(Lower KS statistic and higher p-value indicate more similar distributions)")

    plt.figure(figsize=(12, 6))
    sns.histplot(real_flat, bins=50, color='blue', label='Real Returns', kde=True, stat='density', alpha=0.6)
    sns.histplot(synth_flat, bins=50, color='red', label='TimeGAN Synthetic Returns', kde=True, stat='density', alpha=0.6)
    sns.histplot(gbm_flat, bins=50, color='green', label='GBM Simulated Returns', kde=True, stat='density', alpha=0.6)
    plt.title('Distribution of Daily Close Returns: Real vs. Synthetic vs. GBM')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Return distributions plot saved to {output_path}")

    return real_flat, synth_flat, gbm_flat

def plot_acf_comparison(
    real_flat_returns: np.ndarray,
    synth_flat_returns: np.ndarray,
    gbm_flat_returns: np.ndarray,
    nlags: int = 20,
    output_path: str = 'acf_comparison.png'
):
    """
    Plots Autocorrelation Function (ACF) for returns and absolute returns.

    Args:
        real_flat_returns (np.ndarray): Flattened real returns.
        synth_flat_returns (np.ndarray): Flattened synthetic returns.
        gbm_flat_returns (np.ndarray): Flattened GBM returns.
        nlags (int): Number of lags for ACF calculation.
        output_path (str): File path to save the ACF plots.
    """
    acf_real = compute_acf(real_flat_returns, nlags)
    acf_synth = compute_acf(synth_flat_returns, nlags)
    acf_gbm = compute_acf(gbm_flat_returns, nlags)

    acf_abs_real = compute_acf(np.abs(real_flat_returns), nlags)
    acf_abs_synth = compute_acf(np.abs(synth_flat_returns), nlags)
    acf_abs_gbm = compute_acf(np.abs(gbm_flat_returns), nlags)

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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ACF comparison plots saved to {output_path}")

def plot_tsne_visualization(
    real_sequences_scaled: np.ndarray,
    synthetic_data_scaled: np.ndarray,
    n_viz: int = 500,
    output_path: str = 'tsne_real_vs_synthetic.png'
):
    """
    Performs t-SNE dimensionality reduction and plots the overlap of real and synthetic sequences.

    Args:
        real_sequences_scaled (np.ndarray): Scaled real sequences (N, SEQ_LEN, N_FEATURES).
        synthetic_data_scaled (np.ndarray): Scaled synthetic sequences (N, SEQ_LEN, N_FEATURES).
        n_viz (int): Number of sequences to visualize from each set.
        output_path (str): File path to save the t-SNE plot.
    """
    n_viz = min(n_viz, real_sequences_scaled.shape[0], synthetic_data_scaled.shape[0])

    if n_viz == 0:
        print("Warning: Not enough data for t-SNE visualization. Skipping t-SNE plot.")
        return

    real_sequences_flat_viz = real_sequences_scaled[:n_viz].reshape(n_viz, -1)
    synthetic_sequences_flat_viz = synthetic_data_scaled[:n_viz].reshape(n_viz, -1)

    combined_sequences_flat = np.vstack([real_sequences_flat_viz, synthetic_sequences_flat_viz])

    # Perplexity must be less than n_samples. Minimum of 50 samples usually needed for t-SNE.
    if len(combined_sequences_flat) <= 1:
        print(f"Warning: Insufficient samples ({len(combined_sequences_flat)}) for t-SNE. Need at least 2.")
        return

    perplexity_val = min(30, len(combined_sequences_flat) - 1)
    if perplexity_val < 1:
        print(f"Warning: Perplexity value calculated as {perplexity_val}, which is too low for t-SNE. Skipping t-SNE plot.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300)
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
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"t-SNE visualization plot saved to {output_path}")

def evaluate_trading_strategy(
    real_features_df: pd.DataFrame,
    synthetic_close_returns: np.ndarray,
    strategy_lookback: int = 10,
    annual_risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
    output_path: str = 'synthetic_backtest_distribution.png'
):
    """
    Evaluates a momentum trading strategy on real and synthetic data and plots results.

    Args:
        real_features_df (pd.DataFrame): DataFrame with real historical features.
        synthetic_close_returns (np.ndarray): Synthetic close returns.
        strategy_lookback (int): Lookback period for momentum strategy.
        annual_risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation.
        trading_days_per_year (int): Number of trading days per year.
        output_path (str): File path to save the strategy performance plot.
    """
    real_full_returns = real_features_df['Close'].values
    real_strat_returns = momentum_strategy(real_full_returns, lookback=strategy_lookback)
    real_sharpe_ratio = calculate_sharpe_ratio(real_strat_returns, annual_risk_free_rate, trading_days_per_year)

    print(f"\nStrategy Sharpe Ratio on real historical data: {real_sharpe_ratio:.3f}")

    synth_sharpe_ratios = []
    for i in range(synthetic_close_returns.shape[0]):
        path_returns = synthetic_close_returns[i]
        strat_returns_synth = momentum_strategy(path_returns, lookback=strategy_lookback)
        sharpe = calculate_sharpe_ratio(strat_returns_synth, annual_risk_free_rate, trading_days_per_year)
        synth_sharpe_ratios.append(sharpe)

    synth_sharpe_ratios = np.array(synth_sharpe_ratios)

    print(f"\nStrategy Sharpe Ratio on synthetic data (Mean): {np.mean(synth_sharpe_ratios):.3f}")
    print(f"Strategy Sharpe Ratio on synthetic data (5th Percentile): {np.percentile(synth_sharpe_ratios, 5):.3f}")
    print(f"Strategy Sharpe Ratio on synthetic data (95th Percentile): {np.percentile(synth_sharpe_ratios, 95):.3f}")

    plt.figure(figsize=(10, 6))
    plt.hist(synth_sharpe_ratios, bins=50, alpha=0.7, edgecolor='black', color='coral', label='Synthetic Sharpe Ratios')
    plt.axvline(x=real_sharpe_ratio, color='blue', linestyle='--', linewidth=2, label=f'Real Sharpe: {real_sharpe_ratio:.2f}')
    plt.title('Distribution of Strategy Performance Across Synthetic Paths')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Synthetic backtest distribution plot saved to {output_path}")

# --- Main Orchestration Function ---

def run_financial_gan_synthesis(
    ticker: str = 'SPY',
    start_date: str = '2010-01-01',
    end_date: str = '2024-01-01',
    seq_len: int = 20,
    s0: float = 100.0,
    gbm_n_days_sim: int = 252,
    gbm_n_paths_sim: int = 1000,
    gan_batch_size: int = 128,
    timegan_train_steps: int = 5000,
    custom_gan_epochs: int = 200,
    noise_dim: int = 32,
    layers_dim: int = 128,
    latent_dim: int = 32,
    n_synthetic_samples: int = 1000,
    strategy_lookback: int = 10,
    annual_risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
    nlags_acf: int = 20,
    tsne_n_viz: int = 500,
    output_dir: str = 'outputs'
):
    """
    Main function to run the entire financial GAN synthesis and evaluation pipeline.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).
        seq_len (int): Length of sequences for GAN.
        s0 (float): Initial price for simulations.
        gbm_n_days_sim (int): Number of days for GBM simulation.
        gbm_n_paths_sim (int): Number of paths for GBM simulation.
        gan_batch_size (int): Batch size for GAN training.
        timegan_train_steps (int): Training steps for ydata-synthetic TimeGAN.
        custom_gan_epochs (int): Epochs for custom GAN training.
        noise_dim (int): Noise dimension for GAN generator.
        layers_dim (int): Hidden layer dimension for GAN networks.
        latent_dim (int): Latent dimension for generator.
        n_synthetic_samples (int): Number of synthetic sequences to generate.
        strategy_lookback (int): Lookback period for momentum strategy.
        annual_risk_free_rate (float): Annual risk-free rate.
        trading_days_per_year (int): Trading days per year.
        nlags_acf (int): Number of lags for ACF plots.
        tsne_n_viz (int): Number of samples for t-SNE visualization.
        output_dir (str): Directory to save all plots and outputs.
    """
    print("--- Starting Financial Time Series GAN Synthesis ---")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load and Preprocess Data
    try:
        features_df = load_and_preprocess_data(ticker, start_date, end_date)
    except ValueError as e:
        print(f"Error during data loading: {e}")
        return
    n_features = features_df.shape[1]

    # 2. Run GBM Simulation and Plot
    gbm_returns, gbm_prices = run_gbm_simulation_and_plot(
        features_df=features_df,
        s0=s0,
        n_days_sim=gbm_n_days_sim,
        n_paths_sim=gbm_n_paths_sim,
        output_path=os.path.join(output_dir, 'gbm_sample_paths.png')
    )

    # 3. Prepare Data for GAN Training
    try:
        real_sequences_scaled, scaler = prepare_gan_training_data(features_df, seq_len)
    except ValueError as e:
        print(f"Error during GAN data preparation: {e}")
        return

    # 4. Train GAN Model
    trained_gan_model, d_losses, g_losses = train_gan_model(
        real_sequences=real_sequences_scaled,
        seq_len=seq_len,
        n_features=n_features,
        batch_size=gan_batch_size,
        timegan_train_steps=timegan_train_steps,
        custom_gan_epochs=custom_gan_epochs,
        noise_dim=noise_dim,
        layers_dim=layers_dim,
        latent_dim=latent_dim,
        timegan_available=TIMEGAN_AVAILABLE
    )

    # 5. Plot GAN Losses (if custom GAN was used)
    plot_gan_losses(g_losses, d_losses, output_path=os.path.join(output_dir, 'gan_loss_curves.png'))

    # 6. Generate Synthetic Data
    synthetic_data_original_scale, synthetic_data_scaled_for_tsne = generate_synthetic_data(
        trained_model=trained_gan_model,
        n_samples=n_synthetic_samples,
        seq_len=seq_len,
        n_features=n_features,
        latent_dim=latent_dim,
        scaler=scaler,
        timegan_available=TIMEGAN_AVAILABLE
    )

    # 7. Plot Real vs. Synthetic Price Paths
    synth_close_returns = plot_real_vs_synthetic_paths(
        features_df=features_df,
        synthetic_data_original_scale=synthetic_data_original_scale,
        s0=s0,
        seq_len=seq_len,
        n_samples_to_plot=min(20, n_synthetic_samples),
        output_path=os.path.join(output_dir, 'real_vs_synthetic_paths.png')
    )
    if synth_close_returns.size == 0:
        print("Exiting: No synthetic close returns generated or available for further analysis.")
        return

    # 8. Perform Statistical Comparison
    real_flat_returns, synth_flat_returns, gbm_flat_returns = perform_statistical_comparison(
        real_features_df=features_df,
        synthetic_close_returns=synth_close_returns,
        gbm_returns=gbm_returns,
        output_path=os.path.join(output_dir, 'return_distributions_overlay.png')
    )

    # 9. Plot ACF Comparison
    plot_acf_comparison(
        real_flat_returns=real_flat_returns,
        synth_flat_returns=synth_flat_returns,
        gbm_flat_returns=gbm_flat_returns,
        nlags=nlags_acf,
        output_path=os.path.join(output_dir, 'acf_comparison.png')
    )

    # 10. t-SNE Visualization
    plot_tsne_visualization(
        real_sequences_scaled=real_sequences_scaled,
        synthetic_data_scaled=synthetic_data_scaled_for_tsne,
        n_viz=tsne_n_viz,
        output_path=os.path.join(output_dir, 'tsne_real_vs_synthetic.png')
    )

    # 11. Evaluate Trading Strategy
    evaluate_trading_strategy(
        real_features_df=features_df,
        synthetic_close_returns=synth_close_returns,
        strategy_lookback=strategy_lookback,
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days_per_year=trading_days_per_year,
        output_path=os.path.join(output_dir, 'synthetic_backtest_distribution.png')
    )

    print("\n--- Financial Time Series GAN Synthesis Completed ---")


if __name__ == "__main__":
    # Example usage when running the script directly
    # All plots will be saved in the 'outputs' directory
    run_financial_gan_synthesis(
        ticker='SPY',
        start_date='2010-01-01',
        end_date='2024-01-01',
        seq_len=20,
        gbm_n_paths_sim=1000,
        timegan_train_steps=5000,
        custom_gan_epochs=500, # Increased custom GAN epochs for better results
        n_synthetic_samples=1000,
        output_dir='outputs_SPY_analysis' # Custom output directory for this run
    )
