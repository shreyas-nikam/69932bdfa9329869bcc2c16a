from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# Import all functions and variables from source.py
from source import *  # noqa: F403,F401

# Matplotlib / Seaborn aesthetics (as required by spec)
plt.style.use("ggplot")
sns.set_palette("deep")

# -----------------------------
# Session State Initialization
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home: Overview"

# User inputs
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
if "start_date" not in st.session_state:
    st.session_state.start_date = "2010-01-01"
if "end_date" not in st.session_state:
    st.session_state.end_date = "2024-01-01"

# Data acquisition and preprocessing
if "features" not in st.session_state:
    st.session_state.features = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# GBM Simulation
if "mu_annual" not in st.session_state:
    st.session_state.mu_annual = None
if "sigma_annual" not in st.session_state:
    st.session_state.sigma_annual = None
if "gbm_returns" not in st.session_state:
    st.session_state.gbm_returns = None
if "gbm_prices" not in st.session_state:
    st.session_state.gbm_prices = None
if "gbm_simulated" not in st.session_state:
    st.session_state.gbm_simulated = False

# TimeGAN Training
if "SEQ_LEN" not in st.session_state:
    st.session_state.SEQ_LEN = 20
if "BATCH_SIZE" not in st.session_state:
    st.session_state.BATCH_SIZE = 128
if "TRAIN_STEPS" not in st.session_state:
    st.session_state.TRAIN_STEPS = 5000
if "NOISE_DIM" not in st.session_state:
    st.session_state.NOISE_DIM = 32
if "LAYERS_DIM" not in st.session_state:
    st.session_state.LAYERS_DIM = 128
if "LATENT_DIM" not in st.session_state:
    st.session_state.LATENT_DIM = 32

if "real_sequences" not in st.session_state:
    st.session_state.real_sequences = None
if "TIMEGAN_AVAILABLE" not in st.session_state:
    # TIMEGAN_AVAILABLE is defined in source.py
    st.session_state.TIMEGAN_AVAILABLE = TIMEGAN_AVAILABLE  # noqa: F405
if "synth_model" not in st.session_state:
    st.session_state.synth_model = None
if "discriminator_keras_model" not in st.session_state:
    st.session_state.discriminator_keras_model = None
if "gan_keras_model" not in st.session_state:
    st.session_state.gan_keras_model = None
if "discriminator_losses" not in st.session_state:
    st.session_state.discriminator_losses = []
if "generator_losses" not in st.session_state:
    st.session_state.generator_losses = []
if "timegan_trained" not in st.session_state:
    st.session_state.timegan_trained = False

# Synthetic Data Generation
if "n_samples_synthetic" not in st.session_state:
    st.session_state.n_samples_synthetic = 1000
if "synthetic_data_scaled" not in st.session_state:
    st.session_state.synthetic_data_scaled = None
if "synthetic_data_original_scale" not in st.session_state:
    st.session_state.synthetic_data_original_scale = None
if "synth_close_returns" not in st.session_state:
    st.session_state.synth_close_returns = None
if "synth_prices" not in st.session_state:
    st.session_state.synth_prices = None
if "real_full_returns" not in st.session_state:
    st.session_state.real_full_returns = None
if "real_close_sequences_for_plot" not in st.session_state:
    st.session_state.real_close_sequences_for_plot = None
if "real_prices_for_plot" not in st.session_state:
    st.session_state.real_prices_for_plot = None
if "synthetic_generated" not in st.session_state:
    st.session_state.synthetic_generated = False

# Statistical Quality Assessment
if "real_flat" not in st.session_state:
    st.session_state.real_flat = None
if "synth_flat" not in st.session_state:
    st.session_state.synth_flat = None
if "gbm_flat" not in st.session_state:
    st.session_state.gbm_flat = None
if "ks_gan" not in st.session_state:
    st.session_state.ks_gan = None
if "p_gan" not in st.session_state:
    st.session_state.p_gan = None
if "ks_gbm" not in st.session_state:
    st.session_state.ks_gbm = None
if "p_gbm" not in st.session_state:
    st.session_state.p_gbm = None
if "acf_real" not in st.session_state:
    st.session_state.acf_real = None
if "acf_synth" not in st.session_state:
    st.session_state.acf_synth = None
if "acf_gbm" not in st.session_state:
    st.session_state.acf_gbm = None
if "acf_abs_real" not in st.session_state:
    st.session_state.acf_abs_real = None
if "acf_abs_synth" not in st.session_state:
    st.session_state.acf_abs_synth = None
if "acf_abs_gbm" not in st.session_state:
    st.session_state.acf_abs_gbm = None
if "embedded_sequences" not in st.session_state:
    st.session_state.embedded_sequences = None
if "labels_viz" not in st.session_state:
    st.session_state.labels_viz = None

# Strategy Backtesting
if "STRATEGY_LOOKBACK" not in st.session_state:
    st.session_state.STRATEGY_LOOKBACK = 10
if "real_sharpe_ratio" not in st.session_state:
    st.session_state.real_sharpe_ratio = None
if "synth_sharpe_ratios" not in st.session_state:
    st.session_state.synth_sharpe_ratios = None
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False

# -----------------------------
# Helpers (robust column access)
# -----------------------------
st.title("QuLab: Lab 19 - Synthetic Time-Series Generation for Backtesting & Stress Testing")

st.caption(
    '**Persona:** Sarah, a Quantitative Strategist at "Alpha Horizons Investment Management."')
st.divider()


def _col(df: pd.DataFrame, preferred: str, fallback: str | None = None) -> str:
    """
    Return the actual column name, preferring exact match first, then case-insensitive match,
    then fallback.
    """
    if df is None:
        return preferred
    if preferred in df.columns:
        return preferred
    # case-insensitive search
    low_map = {c.lower(): c for c in df.columns}
    if preferred.lower() in low_map:
        return low_map[preferred.lower()]
    if fallback is not None and fallback in df.columns:
        return fallback
    # last resort: raise a helpful error
    raise KeyError(
        f"Expected column '{preferred}' (or case-insensitive variant) not found. Available: {list(df.columns)}")


# -----------------------------
# App Layout + Navigation
# -----------------------------
st.set_page_config(
    layout="wide", page_title="Synthetic Time-Series Generation")

with st.sidebar:
    # Optional logo (avoid crashing if missing)
    try:
        st.image("https://www.quantuniversity.com/assets/img/logo5.jpg",
                 width='stretch')
    except Exception:
        pass
    st.divider()
    st.title("Navigation")
    page_selection = st.selectbox(
        "Choose a section",
        [
            "Home: Overview",
            "1. Data Acquisition & Preprocessing",
            "2. GBM Simulation (Baseline)",
            "3. TimeGAN Model Training",
            "4. Generate Synthetic Paths",
            "5. Statistical Quality Assessment",
            "6. Strategy Backtesting",
            "7. GAN Limitations & Discussion",
        ],
        key="page_select",
    )
    st.session_state.page = page_selection


# -----------------------------
# Page: Home
# -----------------------------
if st.session_state.page == "Home: Overview":
    st.markdown(
        """
Sarah's firm relies heavily on backtesting strategies and stress testing portfolios. However, she constantly faces the **"single history problem"**:
rare but impactful market events (like the 2008 GFC or 2020 COVID crash) have only occurred once or twice in historical data.
This scarcity limits the robustness of her analyses, making it difficult to assess how strategies truly perform under diverse, unseen, yet plausible market conditions.

Traditional Monte Carlo simulations, based on simple parametric models like **Geometric Brownian Motion (GBM)**, often fail to capture the complex, non-Gaussian statistical properties of real financial data—such as **fat tails**, **volatility clustering**, and **autocorrelation**.
        """
    )
    st.markdown(
        """
**Challenge:** Sarah needs a way to generate new, realistic financial price paths that faithfully mimic the complex statistical characteristics of historical data, going beyond simple parametric assumptions. These synthetic paths will augment the historical record, allowing her to conduct more comprehensive backtesting and stress testing, ultimately leading to more robust strategy development and risk management.
        """
    )
    st.markdown(
        """
**Goal of this Application:** This application will guide Sarah through a practical workflow to generate realistic synthetic financial time series using a **TimeGAN (Generative Adversarial Network for time series)**. She will compare the synthetic data's quality against both real historical data and a traditional **Geometric Brownian Motion (GBM)** baseline. Finally, she will apply a simple investment strategy to these synthetic paths to assess its robustness across a multitude of plausible market scenarios.
        """
    )
    st.markdown("---")
    st.markdown(
        "**Please navigate through the sections using the sidebar to proceed.**")

# -----------------------------
# Page 1: Data Acquisition & Preprocessing
# -----------------------------
elif st.session_state.page == "1. Data Acquisition & Preprocessing":
    st.header("1. Setting Up the Environment and Acquiring Historical Data")
    st.markdown(
        """
Sarah begins by setting up her Python environment and acquiring the necessary historical financial data.
For robust generative modeling, she needs multi-feature (OHLCV) daily data for a representative market index.
        """
    )

    st.subheader("1.1. Acquire and Preprocess Financial Data")
    st.markdown(
        """
Sarah downloads daily OHLCV data for the S&P 500 ETF (AAPL) and transforms it into daily percentage returns and log-transformed volume changes.
This preprocessing is crucial for modeling financial time series, as returns are more stationary than prices.
        """
    )
    st.markdown("The percentage change for prices is calculated as:")
    st.markdown(r"""
$$
\frac{{P_t - P_{{t-1}}}}{{P_{{t-1}}}}
$$""")
    st.markdown(
        r"where $P_t$ is the price at time $t$ and $P_{{t-1}}$ is the price at time $t-1$.")
    st.markdown(
        "For volume, a log-difference transformation is used to stabilize variance:")
    st.markdown(r"""
$$
V_t = \ln(1+V_t) - \ln(1+V_{{t-1}})
$$""")
    st.markdown(r"where $V_t$ is the volume at time $t$.")

    with st.expander("Data Input Parameters"):
        st.session_state.ticker = st.text_input(
            "Financial Ticker (e.g., AAPL):", st.session_state.ticker, key="ticker_input")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.start_date = (
                st.date_input("Start Date:", pd.to_datetime(
                    st.session_state.start_date), key="start_date_input").strftime("%Y-%m-%d")
            )
        with c2:
            st.session_state.end_date = (
                st.date_input("End Date:", pd.to_datetime(
                    st.session_state.end_date), key="end_date_input").strftime("%Y-%m-%d")
            )

    load_clicked = st.button(
        "Load and Preprocess Data", key="load_data_button")
    if load_clicked or st.session_state.data_loaded:
        try:
            if load_clicked or not st.session_state.data_loaded:
                with st.spinner(f"Loading and preprocessing data for {st.session_state.ticker}..."):
                    features = load_and_preprocess_data(  # noqa: F405
                        st.session_state.ticker,
                        st.session_state.start_date,
                        st.session_state.end_date,
                    )

                    # Fit scaler (spec wants scaler stored for later inverse-transform)
                    temp_scaler = MinMaxScaler(feature_range=(-1, 1))
                    temp_scaler.fit(features.values)

                    st.session_state.features = features
                    st.session_state.scaler = temp_scaler
                    st.session_state.data_loaded = True

                    # Reset downstream steps when reloading data
                    st.session_state.gbm_simulated = False
                    st.session_state.timegan_trained = False
                    st.session_state.synthetic_generated = False
                    st.session_state.backtest_run = False

                st.success("Data loaded and preprocessed successfully!")

            if st.session_state.data_loaded and st.session_state.features is not None:
                st.write(
                    f"**Data for {st.session_state.ticker} (first 5 rows):**")
                st.dataframe(st.session_state.features.head(),
                             width='stretch')

                st.write(
                    f"Training data shape: {st.session_state.features.shape[0]} days x {st.session_state.features.shape[1]} features"
                )

                close_col = _col(st.session_state.features, "Close")
                close_returns = st.session_state.features[close_col]

                st.write(
                    f"Close return stats: mean={close_returns.mean():.5f}, std={close_returns.std():.4f}, "
                    f"skew={close_returns.skew():.2f}, kurt={close_returns.kurtosis():.2f}"
                )

                st.subheader("1.2. Explanation of Data Preprocessing")
                st.markdown(
                    """
Sarah knows that financial time series, especially raw prices, are typically non-stationary.
Working with returns stabilizes the data, making it more suitable for modeling.

The `dropna()` step removes initial `NaN` values created by the percentage change calculations, ensuring a clean dataset for training.
The descriptive statistics for 'close' returns provide initial insights into the data's characteristics, especially its mean, standard deviation, skewness, and kurtosis, which are crucial for later comparisons with synthetic data.
                    """
                )

        except ValueError as e:
            st.error(
                f"Error loading data: {e}. Please check the ticker and date range.")
            st.session_state.data_loaded = False
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.data_loaded = False

# -----------------------------
# Page 2: GBM Simulation (Baseline)
# -----------------------------
elif st.session_state.page == "2. GBM Simulation (Baseline)":
    st.header("2. Baseline Model: Geometric Brownian Motion (GBM) Simulation")
    st.markdown(
        """
Before diving into complex generative models, Sarah wants to establish a traditional benchmark.
The **Geometric Brownian Motion (GBM)** model is a fundamental stochastic process widely used in finance to model asset prices.
It assumes that asset returns are normally distributed and exhibit constant drift and volatility.
        """
    )
    st.markdown("The stochastic differential equation for GBM is given by:")
    st.markdown(r"""
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$""")
    st.markdown(r"where $S_t$ is the asset price at time $t$, $\mu$ is the drift coefficient (expected return), $\sigma$ is the volatility coefficient, and $dW_t$ is a Wiener process (standard Brownian motion).")
    st.markdown("Its discrete-time solution for simulating price paths is:")
    st.markdown(
        r"""
$$
S_{{t+1}} = S_t \exp \left[ \left(\mu - \frac{{1}}{{2}}\sigma^2\right)\Delta t + \sigma\sqrt{{\Delta t}} Z_t \right], \quad Z_t \sim \mathcal{{N}}(0,1)
$$""")
    st.markdown(r"Here, $\Delta t$ is the time step (e.g., $1/252$ for daily steps in a year), and $Z_t$ are independent standard normal random variables.")

    if st.session_state.data_loaded and st.session_state.features is not None:
        st.markdown(
            f"Sarah will calibrate GBM parameters ($\\mu$, $\\sigma$) from the historical daily close returns of '{st.session_state.ticker}'."
        )
        st.subheader("2.1. Implement and Calibrate GBM")
        st.markdown(
            """
Sarah implements a function to simulate GBM paths and calibrates its annual drift ($\mu$) and volatility ($\sigma$) using the historical 'close' returns.
            """
        )

        sim_clicked = st.button(
            "Calibrate and Simulate GBM", key="simulate_gbm_button")
        if sim_clicked or st.session_state.gbm_simulated:
            if sim_clicked or not st.session_state.gbm_simulated:
                with st.spinner("Calibrating and simulating GBM paths..."):
                    close_col = _col(st.session_state.features, "Close")
                    historical_close_returns = st.session_state.features[close_col].values

                    mu_annual = float(np.mean(historical_close_returns) * 252)
                    sigma_annual = float(
                        np.std(historical_close_returns) * np.sqrt(252))

                    S0 = 100.0
                    n_days_sim = 252
                    n_paths_sim = 1000

                    gbm_returns, gbm_prices = simulate_gbm(mu_annual, sigma_annual, S0, n_days_sim, n_paths_sim)  # noqa: F405

                    st.session_state.mu_annual = mu_annual
                    st.session_state.sigma_annual = sigma_annual
                    st.session_state.gbm_returns = gbm_returns
                    st.session_state.gbm_prices = gbm_prices
                    st.session_state.gbm_simulated = True

                st.success("GBM calibrated and simulated successfully!")

            if st.session_state.gbm_simulated:
                st.write(
                    f"Calibrated GBM: Annual Mean (mu) = {st.session_state.mu_annual:.4f}, Annual Std (sigma) = {st.session_state.sigma_annual:.4f}")
                st.write(
                    f"GBM simulated paths shape (returns): {st.session_state.gbm_returns.shape}")
                st.write(
                    f"GBM simulated paths shape (prices): {st.session_state.gbm_prices.shape}")

                st.subheader("2.2. Visualize GBM Simulated Paths")
                st.markdown(
                    "Sarah plots a subset of the generated GBM price paths to visually inspect their behavior.")
                fig, ax = plt.subplots(figsize=(12, 6))
                for i in range(min(20, st.session_state.gbm_prices.shape[0])):
                    ax.plot(st.session_state.gbm_prices[i], alpha=0.5)
                ax.set_title(
                    f"Monte Carlo GBM: 20 Sample Price Paths ({st.session_state.ticker})")
                ax.set_xlabel("Trading Day")
                ax.set_ylabel("Price")
                ax.grid(True, linestyle=":", alpha=0.7)
                st.pyplot(fig)

                st.subheader("2.3. Limitations of GBM for Financial Data")
                st.markdown(
                    """
Sarah recognizes that while GBM is a good starting point, its assumptions are often violated by real financial data.
This makes it an insufficient model for robust stress testing and complex strategy backtesting.
                    """
                )
                st.markdown("* **No volatility clustering:** GBM assumes constant volatility ($\\sigma$). Real markets exhibit periods of high and low volatility (GARCH effects), meaning large price movements tend to be followed by large price movements.")
                st.markdown("* **Normal returns:** GBM generates Gaussian (normal) returns. Real financial returns often have \"fat tails\" (excess kurtosis, where $Kurtosis > 3$) and negative skewness, indicating a higher probability of extreme events and larger downside movements than a normal distribution would predict.")
                st.markdown("* **No autocorrelation structure:** GBM returns are independent and identically distributed (i.i.d.). Real returns often show short-term momentum or long-term mean reversion, especially in absolute returns (indicating volatility clustering).")
                st.markdown(
                    "* **No cross-asset dependence dynamics:** Standard GBM models with constant correlation cannot capture the phenomenon of correlation spikes during crises.")
                st.markdown(
                    """
These limitations highlight why Sarah needs a more sophisticated approach like TimeGAN, which can learn and replicate these complex statistical properties directly from data without explicit parametric assumptions.
                    """
                )
        else:
            st.info("Click 'Calibrate and Simulate GBM' to run the simulation.")
    else:
        st.info(
            "Load data first in '1. Data Acquisition & Preprocessing' to enable GBM simulation.")

# -----------------------------
# Page 3: TimeGAN Model Training
# -----------------------------
elif st.session_state.page == "3. TimeGAN Model Training":
    st.header("3. TimeGAN Model: Training for Realistic Time-Series Generation")
    st.markdown(
        """
To overcome the limitations of GBM, Sarah turns to **TimeGAN (Generative Adversarial Network for time series)**.
TimeGAN is specifically designed to capture the complex temporal dependencies and statistical properties inherent in real financial time series.
It employs an adversarial training mechanism, where a `Generator` tries to produce synthetic data indistinguishable from real data, and a `Discriminator` tries to distinguish between real and fake.

Crucially, TimeGAN adds an embedding network, a recovery network, and a supervised loss to ensure it captures not just marginal distributions, but also temporal dynamics like autocorrelation and volatility clustering.
        """
    )
    st.markdown("The overall TimeGAN objective combines three loss components:")
    st.markdown(r"""
$$
\mathcal{{L}}_{{\text{{TimeGAN}}}} = \mathcal{{L}}_{{\text{{reconstruction}}}} + \gamma \mathcal{{L}}_{{\text{{supervised}}}} + \mathcal{{L}}_{{\text{{adversarial}}}}
$$""")
    st.markdown(
        r"where: $ \mathcal{{L}}_{{\text{{reconstruction}}}}$ ensures the autoencoder (embedding and recovery networks) can reconstruct real data from its latent representation; $ \mathcal{{L}}_{{\text{{supervised}}}}$ trains the generator to capture one-step-ahead temporal dynamics in the latent space, guiding it to generate sequences that preserve sequential information; $ \mathcal{{L}}_{{\text{{adversarial}}}}$ is the standard GAN loss, where the discriminator tries to distinguish real vs. synthetic latent sequences, and the generator tries to fool it."
    )

    st.subheader("3.1. Prepare Data for TimeGAN Training")
    st.markdown(
        f"""
TimeGAN requires data to be in sequences (e.g., {st.session_state.SEQ_LEN}-day windows) and typically normalized.
Sarah will create these sequences from her preprocessed OHLCV features and scale them to a range suitable for GAN training (e.g., `[-1, 1]` for `tanh` activation in the generator).
        """
    )

    if st.session_state.data_loaded and st.session_state.features is not None and st.session_state.scaler is not None:
        with st.expander("TimeGAN Hyperparameters"):
            st.session_state.SEQ_LEN = st.slider(
                "Sequence Length (SEQ_LEN):", 5, 60, st.session_state.SEQ_LEN, key="seq_len_slider")
            st.session_state.BATCH_SIZE = st.slider(
                "Batch Size:", 32, 256, st.session_state.BATCH_SIZE, step=32, key="batch_size_slider")
            st.session_state.TRAIN_STEPS = st.slider(
                "Training Steps/Epochs:", 100, 10000, st.session_state.TRAIN_STEPS, step=100, key="train_steps_slider")
            st.session_state.NOISE_DIM = st.slider(
                "Noise Dimension (NOISE_DIM):", 16, 64, st.session_state.NOISE_DIM, key="noise_dim_slider")
            st.session_state.LAYERS_DIM = st.slider(
                "Layers Dimension (LAYERS_DIM):", 64, 256, st.session_state.LAYERS_DIM, key="layers_dim_slider")
            st.session_state.LATENT_DIM = st.slider(
                "Latent Dimension (LATENT_DIM):", 16, 64, st.session_state.LATENT_DIM, key="latent_dim_slider")

        train_clicked = st.button(
            "Prepare and Train TimeGAN", key="train_timegan_button")
        if train_clicked or st.session_state.timegan_trained:
            if train_clicked or not st.session_state.timegan_trained:
                with st.spinner("Preparing data and training TimeGAN (this may take a while for many steps)..."):
                    # Build sequences
                    scaled_features = st.session_state.scaler.transform(
                        st.session_state.features.values)
                    real_sequences = prepare_sequences(scaled_features, st.session_state.SEQ_LEN)  # noqa: F405
                    st.session_state.real_sequences = real_sequences

                    st.write(
                        f"Original features shape: {st.session_state.features.shape}")
                    st.write(f"Scaled features shape: {scaled_features.shape}")
                    st.write(
                        f"Prepared real sequences shape for TimeGAN: {real_sequences.shape}")

                    if real_sequences.size == 0:
                        st.error(
                            "Real sequences are empty. Cannot train TimeGAN. Adjust SEQ_LEN or data range.")
                        st.session_state.timegan_trained = False
                    else:
                        st.session_state.TIMEGAN_AVAILABLE = TIMEGAN_AVAILABLE  # noqa: F405

                        # ---- Preferred path: ydata-synthetic TimeGAN ----
                        if st.session_state.TIMEGAN_AVAILABLE:
                            st.markdown(
                                "Training TimeGAN using `ydata-synthetic`...")

                            # ModelParameters / TimeGAN are available only if ydata-synthetic imported in source.py
                            gan_args = ModelParameters(  # noqa: F405
                                batch_size=st.session_state.BATCH_SIZE,
                                lr=5e-4,
                                noise_dim=st.session_state.NOISE_DIM,
                                layers_dim=st.session_state.LAYERS_DIM,
                                latent_dim=st.session_state.LATENT_DIM,
                            )
                            synth = TimeGAN(  # noqa: F405
                                model_parameters=gan_args,
                                n_seq=st.session_state.SEQ_LEN,
                                n_features=st.session_state.features.shape[1],
                                gamma=1,
                            )
                            synth.train(
                                real_sequences, train_steps=st.session_state.TRAIN_STEPS)
                            st.session_state.synth_model = synth
                            st.markdown(
                                "TimeGAN training completed using `ydata-synthetic`.")

                        # ---- Fallback path: custom simplified GAN ----
                        else:
                            st.warning(
                                "ydata-synthetic not installed or failed to import; using custom GAN implementation.")
                            st.markdown(
                                "Building and training custom simplified GAN...")

                            n_features = st.session_state.features.shape[1]
                            generator_model = build_generator(  # noqa: F405
                                st.session_state.LATENT_DIM,
                                st.session_state.SEQ_LEN,
                                n_features,
                                st.session_state.LAYERS_DIM,
                            )
                            discriminator_model = build_discriminator(  # noqa: F405
                                st.session_state.SEQ_LEN,
                                n_features,
                                st.session_state.LAYERS_DIM,
                            )

                            discriminator_model.compile(
                                optimizer=Adam(
                                    learning_rate=0.0002, beta_1=0.5),
                                loss="binary_crossentropy",
                                metrics=["accuracy"],
                            )

                            discriminator_model.trainable = False
                            z_input = tf.keras.Input(
                                shape=(st.session_state.LATENT_DIM,))
                            fake_seq = generator_model(z_input)
                            validity = discriminator_model(fake_seq)

                            gan_model = Model(z_input, validity, name="GAN")
                            gan_model.compile(optimizer=Adam(
                                learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

                            epochs = int(st.session_state.TRAIN_STEPS)
                            d_losses, g_losses = [], []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for epoch in range(epochs):
                                idx = np.random.randint(
                                    0, real_sequences.shape[0], st.session_state.BATCH_SIZE)
                                real_batch = real_sequences[idx]

                                noise = np.random.normal(
                                    0, 1, (st.session_state.BATCH_SIZE, st.session_state.LATENT_DIM))
                                fake_batch = generator_model.predict(
                                    noise, verbose=0)

                                d_loss_real = discriminator_model.train_on_batch(
                                    real_batch, np.ones((st.session_state.BATCH_SIZE, 1)) * 0.9)
                                d_loss_fake = discriminator_model.train_on_batch(
                                    fake_batch, np.zeros((st.session_state.BATCH_SIZE, 1)) * 0.1)
                                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                                noise = np.random.normal(
                                    0, 1, (st.session_state.BATCH_SIZE, st.session_state.LATENT_DIM))
                                g_loss = gan_model.train_on_batch(
                                    noise, np.ones((st.session_state.BATCH_SIZE, 1)))

                                # Extract loss values handling 0-dimensional arrays
                                if isinstance(d_loss, np.ndarray):
                                    d_losses.append(
                                        float(d_loss.item() if d_loss.ndim == 0 else d_loss[0]))
                                elif isinstance(d_loss, (list, tuple)):
                                    d_losses.append(float(d_loss[0]))
                                else:
                                    d_losses.append(float(d_loss))

                                if isinstance(g_loss, np.ndarray):
                                    g_losses.append(
                                        float(g_loss.item() if g_loss.ndim == 0 else g_loss[0]))
                                elif isinstance(g_loss, (list, tuple)):
                                    g_losses.append(float(g_loss[0]))
                                else:
                                    g_losses.append(float(g_loss))

                                progress_bar.progress((epoch + 1) / epochs)
                                if epoch % (epochs // 10 if epochs // 10 > 0 else 1) == 0 or epoch == epochs - 1:
                                    status_text.text(
                                        f"Epoch {epoch+1}/{epochs}: D loss={d_losses[-1]:.4f}, G loss={g_losses[-1]:.4f}")

                            st.session_state.synth_model = generator_model
                            st.session_state.discriminator_keras_model = discriminator_model
                            st.session_state.gan_keras_model = gan_model
                            st.session_state.discriminator_losses = d_losses
                            st.session_state.generator_losses = g_losses
                            st.markdown("Custom GAN training completed.")

                        st.session_state.timegan_trained = True
                        st.success("TimeGAN training completed!")

            if st.session_state.timegan_trained:
                st.subheader("3.2. Explanation of TimeGAN Training")
                st.markdown(
                    """
Sarah understands that training GANs can be challenging, often exhibiting instability or mode collapse.
The `ydata-synthetic` library simplifies this by encapsulating the complex TimeGAN architecture (embedding, generator, discriminator, recovery networks) and its specialized loss functions.

The `train_steps` parameter dictates the duration of adversarial training, a critical factor for achieving high-quality synthetic data.
If the custom GAN is used, the label smoothing (real labels `0.9`, fake `0.1` or `0.0`) is a common technique to stabilize training and prevent the discriminator from becoming too strong too quickly.
                    """
                )

                if (not st.session_state.TIMEGAN_AVAILABLE) and st.session_state.generator_losses:
                    st.subheader("3.3. GAN Training Loss Curves")
                    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
                    ax_loss.plot(st.session_state.generator_losses,
                                 label="Generator Loss", alpha=0.8)
                    ax_loss.plot(st.session_state.discriminator_losses,
                                 label="Discriminator Loss", alpha=0.8)
                    ax_loss.set_title("GAN Training Losses over Epochs")
                    ax_loss.set_xlabel("Epoch")
                    ax_loss.set_ylabel("Loss")
                    ax_loss.legend()
                    ax_loss.grid(True, linestyle=":", alpha=0.7)
                    st.pyplot(fig_loss)
        else:
            st.info(
                "Click 'Prepare and Train TimeGAN' to begin training. This might take a while.")
    else:
        st.info(
            "Load data first in '1. Data Acquisition & Preprocessing' to enable TimeGAN training.")

# -----------------------------
# Page 4: Generate Synthetic Paths
# -----------------------------
elif st.session_state.page == "4. Generate Synthetic Paths":
    st.header("4. Generating Synthetic Financial Paths")
    st.markdown(
        f"""
With the TimeGAN model trained, Sarah can now generate an ensemble of new synthetic financial time-series paths.
This is the core output of the generative model, providing the augmented data needed for robust backtesting and stress testing.

She aims to generate 1,000 synthetic sequences, each mirroring the length and features of her real historical sequences (e.g., {st.session_state.SEQ_LEN} days, OHLCV).
        """
    )

    st.subheader("4.1. Generate Synthetic Sequences")
    st.markdown(
        """
Sarah uses the trained TimeGAN to generate a large number of synthetic sequences.
These sequences are still in their scaled, multi-feature format.
        """
    )

    if st.session_state.timegan_trained and st.session_state.features is not None and st.session_state.scaler is not None:
        st.session_state.n_samples_synthetic = st.slider(
            "Number of Synthetic Paths to Generate:",
            100,
            5000,
            st.session_state.n_samples_synthetic,
            step=100,
            key="num_synth_paths_slider",
        )

        gen_clicked = st.button(
            "Generate Synthetic Paths", key="generate_synthetic_button")
        if gen_clicked or st.session_state.synthetic_generated:
            if gen_clicked or not st.session_state.synthetic_generated:
                with st.spinner(f"Generating {st.session_state.n_samples_synthetic} synthetic paths..."):
                    if st.session_state.TIMEGAN_AVAILABLE:
                        synthetic_data_scaled = st.session_state.synth_model.sample(
                            n_samples=st.session_state.n_samples_synthetic)
                    else:
                        noise_for_generation = np.random.normal(
                            0, 1, (st.session_state.n_samples_synthetic,
                                   st.session_state.LATENT_DIM)
                        )
                        synthetic_data_scaled = st.session_state.synth_model.predict(
                            noise_for_generation, verbose=0)

                    st.session_state.synthetic_data_scaled = synthetic_data_scaled
                    st.write(
                        f"Generated synthetic data shape (scaled): {synthetic_data_scaled.shape}")

                    # Inverse transform to original feature space
                    n_features = st.session_state.features.shape[1]
                    dummy_features = np.zeros(
                        (st.session_state.n_samples_synthetic * st.session_state.SEQ_LEN, n_features))
                    dummy_features[:, :] = synthetic_data_scaled.reshape(
                        st.session_state.n_samples_synthetic * st.session_state.SEQ_LEN, n_features)
                    synthetic_data_original_scale = st.session_state.scaler.inverse_transform(dummy_features).reshape(
                        st.session_state.n_samples_synthetic, st.session_state.SEQ_LEN, n_features
                    )

                    # Extract close-returns feature (source.py builds Open/High/Low/Close in that order)
                    # Index 3 corresponds to Close if columns are [Open, High, Low, Close, volume]
                    synth_close_returns = synthetic_data_original_scale[:, :, 3]

                    # Convert to price paths for visualization
                    synth_prices = np.array(
                        [100 * np.exp(np.cumsum(path_returns)) for path_returns in synth_close_returns])
                    synth_prices = np.column_stack(
                        [np.full(st.session_state.n_samples_synthetic, 100.0), synth_prices])

                    st.session_state.synthetic_data_original_scale = synthetic_data_original_scale
                    st.session_state.synth_close_returns = synth_close_returns
                    st.session_state.synth_prices = synth_prices

                    st.write(
                        f"Synthetic close returns shape: {synth_close_returns.shape}")
                    st.write(f"Synthetic prices shape: {synth_prices.shape}")

                    # Prepare real historical close returns for plotting comparison
                    close_col = _col(st.session_state.features, "Close")
                    st.session_state.real_full_returns = st.session_state.features[close_col].values

                    num_real_paths_to_plot = min(
                        st.session_state.n_samples_synthetic,
                        max(
                            0, st.session_state.real_full_returns.shape[0] - st.session_state.SEQ_LEN + 1),
                    )

                    real_close_sequences_for_plot_raw = prepare_sequences(  # noqa: F405
                        st.session_state.real_full_returns.reshape(-1, 1),
                        st.session_state.SEQ_LEN,
                    )

                    if real_close_sequences_for_plot_raw.size > 0 and num_real_paths_to_plot > 0:
                        real_close_sequences_for_plot = real_close_sequences_for_plot_raw.squeeze()[
                            :num_real_paths_to_plot]
                        if real_close_sequences_for_plot.ndim == 1:
                            real_close_sequences_for_plot = real_close_sequences_for_plot.reshape(
                                num_real_paths_to_plot, -1)

                        real_prices_for_plot = np.array(
                            [100 * np.exp(np.cumsum(path_returns)) for path_returns in real_close_sequences_for_plot])
                        real_prices_for_plot = np.column_stack(
                            [np.full(num_real_paths_to_plot, 100.0), real_prices_for_plot])
                    else:
                        real_close_sequences_for_plot = np.array([])
                        real_prices_for_plot = np.array([])
                        st.warning(
                            "Not enough real data to create sequences for plotting.")

                    st.session_state.real_close_sequences_for_plot = real_close_sequences_for_plot
                    st.session_state.real_prices_for_plot = real_prices_for_plot

                    st.session_state.synthetic_generated = True

                st.success(
                    "Synthetic paths generated and processed successfully!")

            if st.session_state.synthetic_generated:
                st.subheader("4.2. Visualize Real vs. Synthetic Paths")
                st.markdown(
                    """
Sarah visually inspects a subset of the generated TimeGAN price paths alongside real historical paths.
This initial qualitative check helps confirm if the synthetic data "looks" realistic.
                    """
                )

                fig_paths, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Real paths
                num_real_paths_to_plot_viz = min(
                    20,
                    st.session_state.real_prices_for_plot.shape[0] if isinstance(
                        st.session_state.real_prices_for_plot, np.ndarray) else 0,
                )
                if num_real_paths_to_plot_viz > 0:
                    for i in range(num_real_paths_to_plot_viz):
                        ax1.plot(
                            st.session_state.real_prices_for_plot[i], alpha=0.4)
                    ax1.set_title(
                        f"Real {st.session_state.SEQ_LEN}-Day Return Paths ({st.session_state.ticker})")
                    ax1.set_xlabel("Trading Day")
                    ax1.set_ylabel("Cumulative Price")
                    ax1.grid(True, linestyle=":", alpha=0.7)
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "Not enough real data for plotting",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax1.transAxes,
                    )
                    ax1.set_title("Real Paths")

                # Synthetic paths
                if st.session_state.synth_prices is not None and st.session_state.synth_prices.shape[0] > 0:
                    for i in range(min(20, st.session_state.synth_prices.shape[0])):
                        ax2.plot(st.session_state.synth_prices[i], alpha=0.4)
                    ax2.set_title(
                        f"TimeGAN Synthetic {st.session_state.SEQ_LEN}-Day Return Paths")
                    ax2.set_xlabel("Trading Day")
                    ax2.set_ylabel("Cumulative Price")
                    ax2.grid(True, linestyle=":", alpha=0.7)
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No synthetic data for plotting",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax2.transAxes,
                    )
                    ax2.set_title("Synthetic Paths")

                plt.suptitle(
                    "Real vs. Synthetic Financial Time Series", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                st.pyplot(fig_paths)

                st.subheader("4.3. Explanation of Synthetic Path Generation")
                st.markdown(
                    """
The generation process transforms random noise into structured financial time series.
The visual comparison is a quick sanity check.

While the synthetic paths should mimic the overall pattern of real paths (e.g., general trends, volatility levels), they won't be identical, which is the point: generating *new, plausible* scenarios.

This visual validation is a precursor to more rigorous statistical assessment.
                    """
                )
        else:
            st.info("Click 'Generate Synthetic Paths' to proceed.")
    else:
        st.info(
            "Please complete TimeGAN training in '3. TimeGAN Model Training' to generate synthetic paths.")

# -----------------------------
# Page 5: Statistical Quality Assessment
# -----------------------------
elif st.session_state.page == "5. Statistical Quality Assessment":
    st.header("5. Statistical Quality Assessment of Synthetic Data")
    st.markdown(
        """
Visual inspection is not enough. Sarah needs to quantitatively assess how well the synthetic data captures the key statistical properties and "stylized facts" of real financial returns, especially comparing TimeGAN against the GBM baseline.
This rigorous evaluation directly addresses the limitations of traditional models and verifies the value of the generative approach.
        """
    )

    st.subheader("5.1. Compare Descriptive Statistics")
    st.markdown(
        """
Sarah computes the mean, standard deviation, skewness, and kurtosis for real, TimeGAN-generated, and GBM-simulated close returns.
She's looking for TimeGAN to better match the fat tails (high kurtosis, $Kurtosis > 3$) and negative skewness often observed in real financial data, which GBM typically misses.
        """
    )

    if (
        st.session_state.synthetic_generated
        and st.session_state.gbm_simulated
        and st.session_state.real_full_returns is not None
        and st.session_state.synth_close_returns is not None
        and st.session_state.gbm_returns is not None
    ):
        # Flatten arrays for statistics and KS test (ensure consistent length for GBM)
        st.session_state.real_flat = st.session_state.real_full_returns
        st.session_state.synth_flat = st.session_state.synth_close_returns.flatten()
        st.session_state.gbm_flat = st.session_state.gbm_returns.flatten()[
            : len(st.session_state.real_flat)]

        assess_clicked = st.button(
            "Perform Statistical Assessment", key="assess_quality_button")
        if assess_clicked:
            st.markdown("---")
            st.markdown("### Descriptive Statistics Comparison:")

            metrics_data = []
            metrics_to_compare = [
                ("Mean", np.mean),
                ("Std Dev", np.std),
                ("Skewness", lambda x: float(describe(x).skewness) if np.size(x) > 1 else np.nan),  # noqa: F405
                ("Kurtosis", lambda x: float(describe(x).kurtosis) if np.size(x) > 1 else np.nan),  # noqa: F405
            ]
            for name, func in metrics_to_compare:
                r_val = func(st.session_state.real_flat)
                s_val = func(st.session_state.synth_flat)
                g_val = func(st.session_state.gbm_flat)
                metrics_data.append(
                    {
                        "Metric": name,
                        "Real": f"{r_val:.5f}",
                        "Synthetic (TimeGAN)": f"{s_val:.5f}",
                        "GBM": f"{g_val:.5f}",
                    }
                )
            st.table(pd.DataFrame(metrics_data))

            st.subheader(
                "5.2. Kolmogorov-Smirnov (KS) Test for Distributional Similarity")
            st.markdown(
                """
The **Kolmogorov-Smirnov (KS) test** quantifies the maximum difference between the cumulative distribution functions (CDFs) of two samples.
A lower KS statistic and a higher p-value (typically $p > 0.05$) indicate that we cannot reject the null hypothesis that the two samples are drawn from the same underlying distribution.

Sarah uses this to compare the distributional similarity of synthetic returns to real returns.
                """
            )
            st.markdown("The KS statistic $D_{KS}$ is defined as:")
            st.markdown(
                r"""
$$
D_{{KS}} = \sup_x |F_{{\text{{real}}}}(x) - F_{{\text{{synth}}}}(x)|
$$""")
            st.markdown(
                r"where $F_{{\text{{real}}}}(x)$ and $F_{{\text{{synth}}}}(x)$ are the empirical cumulative distribution functions of the real and synthetic data, respectively.")

            ks_gan, p_gan = ks_2samp(st.session_state.real_flat, st.session_state.synth_flat)  # noqa: F405
            ks_gbm, p_gbm = ks_2samp(st.session_state.real_flat, st.session_state.gbm_flat)  # noqa: F405

            st.session_state.ks_gan, st.session_state.p_gan = ks_gan, p_gan
            st.session_state.ks_gbm, st.session_state.p_gbm = ks_gbm, p_gbm

            st.markdown("**Kolmogorov-Smirnov Test Results:**")
            st.write(f"TimeGAN vs Real: KS Stat={ks_gan:.4f} (p={p_gan:.4f})")
            st.write(f"GBM vs Real: KS Stat={ks_gbm:.4f} (p={p_gbm:.4f})")
            st.markdown(
                "(Lower KS statistic and higher p-value indicate more similar distributions)")

            st.subheader("5.3. Compare Return Distributions (Histograms)")
            st.markdown(
                """
Sarah generates histograms to visually overlay the return distributions for real, TimeGAN, and GBM data.
This visualization complements the descriptive statistics and KS test, allowing her to visually confirm the capture of fat tails.
                """
            )

            fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
            sns.histplot(st.session_state.real_flat, bins=50, label="Real Returns",
                         kde=True, stat="density", alpha=0.6, ax=ax_hist)
            sns.histplot(st.session_state.synth_flat, bins=50, label="TimeGAN Synthetic Returns",
                         kde=True, stat="density", alpha=0.6, ax=ax_hist)
            sns.histplot(st.session_state.gbm_flat, bins=50, label="GBM Simulated Returns",
                         kde=True, stat="density", alpha=0.6, ax=ax_hist)
            ax_hist.set_title(
                "Distribution of Daily Close Returns: Real vs. Synthetic vs. GBM")
            ax_hist.set_xlabel("Daily Return")
            ax_hist.set_ylabel("Density")
            ax_hist.legend()
            ax_hist.grid(True, linestyle=":", alpha=0.7)
            st.pyplot(fig_hist)

            st.subheader("5.4. Autocorrelation Function (ACF) Comparison")
            st.markdown(
                """
Volatility clustering, where periods of high volatility are followed by high volatility, is a critical stylized fact.
This often manifests as significant autocorrelation in the absolute returns.

GBM, by construction, has no autocorrelation. Sarah computes and compares the Autocorrelation Function (ACF) for real, TimeGAN, and GBM returns (and absolute returns) up to a specified lag.
                """
            )

            nlags = 20
            st.session_state.acf_real = compute_acf(st.session_state.real_flat, nlags)  # noqa: F405
            st.session_state.acf_synth = compute_acf(st.session_state.synth_flat, nlags)  # noqa: F405
            st.session_state.acf_gbm = compute_acf(st.session_state.gbm_flat, nlags)  # noqa: F405

            st.session_state.acf_abs_real = compute_acf(np.abs(st.session_state.real_flat), nlags)  # noqa: F405
            st.session_state.acf_abs_synth = compute_acf(np.abs(st.session_state.synth_flat), nlags)  # noqa: F405
            st.session_state.acf_abs_gbm = compute_acf(np.abs(st.session_state.gbm_flat), nlags)  # noqa: F405

            fig_acf, (ax_acf1, ax_acf2) = plt.subplots(1, 2, figsize=(16, 6))
            ax_acf1.plot(range(len(st.session_state.acf_real)), st.session_state.acf_real,
                         label="Real Returns", marker="o", linestyle="--")
            ax_acf1.plot(range(len(st.session_state.acf_synth)), st.session_state.acf_synth,
                         label="TimeGAN Synthetic Returns", marker="x", linestyle="-.")
            ax_acf1.plot(range(len(st.session_state.acf_gbm)), st.session_state.acf_gbm,
                         label="GBM Simulated Returns", marker="^", linestyle=":")
            ax_acf1.set_title("Autocorrelation Function (ACF) of Returns")
            ax_acf1.set_xlabel("Lag")
            ax_acf1.set_ylabel("Autocorrelation")
            ax_acf1.grid(True, linestyle=":", alpha=0.7)
            ax_acf1.legend()

            ax_acf2.plot(range(len(st.session_state.acf_abs_real)), st.session_state.acf_abs_real,
                         label="Real Absolute Returns", marker="o", linestyle="--")
            ax_acf2.plot(range(len(st.session_state.acf_abs_synth)), st.session_state.acf_abs_synth,
                         label="TimeGAN Synthetic Absolute Returns", marker="x", linestyle="-.")
            ax_acf2.plot(range(len(st.session_state.acf_abs_gbm)), st.session_state.acf_abs_gbm,
                         label="GBM Simulated Absolute Returns", marker="^", linestyle=":")
            ax_acf2.set_title(
                "Autocorrelation Function (ACF) of Absolute Returns (Volatility Clustering)")
            ax_acf2.set_xlabel("Lag")
            ax_acf2.set_ylabel("Autocorrelation")
            ax_acf2.grid(True, linestyle=":", alpha=0.7)
            ax_acf2.legend()

            plt.tight_layout()
            st.pyplot(fig_acf)

            st.subheader("5.5. t-SNE Visualization for Manifold Overlap")
            st.markdown(
                """
**t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a dimensionality reduction technique particularly effective for visualizing high-dimensional data in 2D or 3D.

Sarah uses t-SNE to project the real and synthetic time sequences into a 2D space.
Significant overlap between the real and synthetic data clusters indicates that the TimeGAN has learned to generate data that resides on the same manifold as the real data, suggesting high fidelity and realism.

Separation, conversely, might indicate mode collapse or a distributional mismatch.
                """
            )

            n_viz = min(
                500,
                st.session_state.real_sequences.shape[0] if st.session_state.real_sequences is not None else 0,
                st.session_state.synthetic_data_scaled.shape[
                    0] if st.session_state.synthetic_data_scaled is not None else 0,
            )

            if n_viz > 0:
                real_sequences_flat_viz = st.session_state.real_sequences[:n_viz].reshape(
                    n_viz, -1)
                synthetic_sequences_flat_viz = st.session_state.synthetic_data_scaled[:n_viz].reshape(
                    n_viz, -1)
                combined_sequences_flat = np.vstack(
                    [real_sequences_flat_viz, synthetic_sequences_flat_viz])
                labels_viz = ["Real"] * n_viz + ["Synthetic"] * n_viz

                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=30,
                    max_iter=300,
                    init="pca",
                    learning_rate="auto",
                )
                embedded_sequences = tsne.fit_transform(
                    combined_sequences_flat)

                st.session_state.embedded_sequences = embedded_sequences
                st.session_state.labels_viz = labels_viz

                fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
                ax_tsne.scatter(
                    embedded_sequences[:n_viz, 0], embedded_sequences[:n_viz, 1], alpha=0.5, s=20, label="Real")
                ax_tsne.scatter(
                    embedded_sequences[n_viz:, 0], embedded_sequences[n_viz:, 1], alpha=0.5, s=20, label="Synthetic")
                ax_tsne.set_title(
                    "t-SNE: Real vs. Synthetic Return Sequences Manifold Overlap")
                ax_tsne.set_xlabel("t-SNE Component 1")
                ax_tsne.set_ylabel("t-SNE Component 2")
                ax_tsne.legend(fontsize=12)
                ax_tsne.grid(True, linestyle=":", alpha=0.7)
                st.pyplot(fig_tsne)
            else:
                st.warning(
                    "Not enough real or synthetic sequences available for t-SNE visualization. Please ensure data is loaded and TimeGAN is trained.")

            st.subheader("5.6. Explanation of Statistical Quality Assessment")
            st.markdown(
                """
This comprehensive assessment confirms TimeGAN's ability to capture financial stylized facts beyond what GBM can.
Sarah can now confidently present her findings:
                """
            )
            st.markdown("* **Descriptive Statistics & Histograms:** TimeGAN should show kurtosis values closer to real data (indicating fat tails) and potentially better skewness than GBM, which defaults to normal-like distributions.")
            st.markdown("* **KS Test:** A lower KS statistic and higher p-value for TimeGAN compared to GBM suggest that TimeGAN's generated distribution is statistically more similar to the real data.")
            st.markdown("* **ACF Plots:** TimeGAN should capture autocorrelation in returns (especially in short lags for real data) and, crucially, in absolute returns (indicating volatility clustering), which GBM will fail to replicate.")
            st.markdown('* **t-SNE:** Significant overlap indicates that the synthetic data points occupy the same underlying data space or "manifold" as the real data, implying high-fidelity generation.')
            st.markdown(
                "This quantitative evidence is critical for justifying the use of TimeGAN-generated data for subsequent analysis steps.")
        else:
            st.info(
                "Click 'Perform Statistical Assessment' to view the quality metrics.")
    else:
        st.info("Please generate synthetic paths (Section 4) and simulate GBM (Section 2) to perform statistical assessment.")

# -----------------------------
# Page 6: Strategy Backtesting
# -----------------------------
elif st.session_state.page == "6. Strategy Backtesting":
    st.header("6. Strategy Backtesting on Synthetic Data")
    st.markdown(
        """
The ultimate goal for Sarah is to use the generated synthetic paths to rigorously backtest investment strategies.
Instead of a single point estimate of performance from one historical path, she can now obtain a *distribution* of performance metrics (e.g., Sharpe ratios) across thousands of plausible market scenarios.

This provides a more robust understanding of a strategy's true performance and its resilience to varying market conditions, directly addressing concerns about backtest overfitting.
        """
    )

    st.subheader("6.1. Implement a Simple Momentum Strategy")
    st.markdown(
        """
Sarah defines a simple rule-based momentum strategy: go long if the sum of returns over a `lookback` period is positive.
        """
    )
    st.markdown("The annualized Sharpe Ratio is calculated as:")
    st.markdown(
        r"""
$$
\text{Sharpe Ratio} = \frac{\text{mean(returns)}}{\text{std(returns)}} \times \sqrt{252}
$$""")
    st.markdown(
        r"where $\text{mean(returns)}$ is the average daily return, $\text{std(returns)}$ is the standard deviation of daily returns, and $\sqrt{252}$ annualizes the metric based on 252 trading days per year."
    )

    if st.session_state.synthetic_generated and st.session_state.real_full_returns is not None:
        st.session_state.STRATEGY_LOOKBACK = st.slider(
            "Momentum Strategy Lookback Period (days):",
            5,
            60,
            st.session_state.STRATEGY_LOOKBACK,
            key="strategy_lookback_slider",
        )

        bt_clicked = st.button("Run Strategy Backtest",
                               key="run_backtest_button")
        if bt_clicked or st.session_state.backtest_run:
            if bt_clicked or not st.session_state.backtest_run:
                with st.spinner("Backtesting strategy on real and synthetic data..."):
                    # Backtest on Real Historical Data
                    real_strat_returns = momentum_strategy(st.session_state.real_full_returns, lookback=st.session_state.STRATEGY_LOOKBACK)  # noqa: F405
                    real_sharpe_ratio = calculate_sharpe_ratio(real_strat_returns)  # noqa: F405
                    st.session_state.real_sharpe_ratio = real_sharpe_ratio
                    st.markdown(
                        f"**Strategy Sharpe Ratio on real historical data: {real_sharpe_ratio:.3f}**")

                    # Backtest on Synthetic Data Ensemble
                    synth_sharpe_ratios = []
                    if st.session_state.synth_close_returns is not None and st.session_state.synth_close_returns.shape[0] > 0:
                        for i in range(st.session_state.synth_close_returns.shape[0]):
                            path_returns = st.session_state.synth_close_returns[i]
                            strat_returns_synth = momentum_strategy(path_returns, lookback=st.session_state.STRATEGY_LOOKBACK)  # noqa: F405
                            sharpe = calculate_sharpe_ratio(strat_returns_synth)  # noqa: F405
                            synth_sharpe_ratios.append(sharpe)
                        st.session_state.synth_sharpe_ratios = np.array(
                            synth_sharpe_ratios)

                        st.markdown(
                            f"**Strategy Sharpe Ratio on synthetic data (Mean): {np.mean(st.session_state.synth_sharpe_ratios):.3f}**")
                        st.markdown(
                            f"**Strategy Sharpe Ratio on synthetic data (5th Percentile): {np.percentile(st.session_state.synth_sharpe_ratios, 5):.3f}**")
                        st.markdown(
                            f"**Strategy Sharpe Ratio on synthetic data (95th Percentile): {np.percentile(st.session_state.synth_sharpe_ratios, 95):.3f}**")
                    else:
                        st.warning(
                            "No synthetic data available for backtesting.")
                        st.session_state.synth_sharpe_ratios = np.array([])

                    st.session_state.backtest_run = True

                st.success("Strategy backtesting completed!")

            if st.session_state.backtest_run:
                st.subheader(
                    "6.2. Visualize Distribution of Synthetic Sharpe Ratios")
                st.markdown(
                    """
Sarah visualizes the distribution of Sharpe ratios from the synthetic backtests using a histogram.
She also marks the real historical Sharpe ratio on this histogram for comparison.

This visualization is the key deliverable for assessing strategy robustness.
                    """
                )

                if st.session_state.synth_sharpe_ratios is not None and st.session_state.synth_sharpe_ratios.size > 0:
                    fig_sharpe, ax_sharpe = plt.subplots(figsize=(10, 6))
                    ax_sharpe.hist(st.session_state.synth_sharpe_ratios, bins=50,
                                   alpha=0.7, edgecolor="black", label="Synthetic Sharpe Ratios")
                    if st.session_state.real_sharpe_ratio is not None:
                        ax_sharpe.axvline(
                            x=st.session_state.real_sharpe_ratio,
                            linestyle="--",
                            linewidth=2,
                            label=f"Real Sharpe: {st.session_state.real_sharpe_ratio:.2f}",
                        )
                    ax_sharpe.set_title(
                        "Distribution of Strategy Performance Across Synthetic Paths")
                    ax_sharpe.set_xlabel("Sharpe Ratio")
                    ax_sharpe.set_ylabel("Frequency")
                    ax_sharpe.legend()
                    ax_sharpe.grid(True, linestyle=":", alpha=0.7)
                    st.pyplot(fig_sharpe)
                else:
                    st.warning(
                        "No synthetic Sharpe ratios to plot. Ensure synthetic paths were generated and backtesting ran successfully.")

                st.subheader(
                    "6.3. Explanation of Synthetic Backtest Distribution")
                st.markdown(
                    """
This distribution of Sharpe ratios is the financial centerpiece for Sarah.
Instead of a single point estimate, she now has a confidence interval for the strategy's performance.
                    """
                )
                st.markdown(
                    "* If the real historical Sharpe ratio falls within the central part of the synthetic distribution, it suggests the strategy is robust and its historical performance is plausible across a range of scenarios."
                )
                st.markdown(
                    "* If the real Sharpe ratio is an extreme outlier (e.g., above the 95th percentile), it could indicate that the strategy's historical performance was fortuitous or that the backtest is overfitted to the single historical path."
                )
                st.markdown(
                    """
This approach significantly enhances Sarah's ability to evaluate strategy robustness, informing whether to deploy or refine the strategy.
It moves beyond "what happened" to "what could happen."
                    """
                )
        else:
            st.info(
                "Click 'Run Strategy Backtest' to see the performance distribution.")
    else:
        st.info(
            "Please generate synthetic paths (Section 4) to run strategy backtesting.")

# -----------------------------
# Page 7: GAN Limitations & Discussion
# -----------------------------
elif st.session_state.page == "7. GAN Limitations & Discussion":
    st.header("7. Addressing GAN Limitations: Mode Collapse and Training Stability")
    st.markdown(
        """
While powerful, GANs, including TimeGAN, come with inherent challenges that Sarah, as a quantitative strategist, must be aware of.
Understanding these limitations is crucial for correctly interpreting results and troubleshooting.
        """
    )

    st.subheader(
        "7.1. Discussion: Mode Collapse, Training Instability, and Hyperparameter Sensitivity")
    st.markdown("* **Mode Collapse:** This is a primary GAN failure mode. It occurs when the generator learns to produce only a limited variety of samples, essentially \"collapsing\" to a few modes of the real data distribution.")
    st.markdown(" * **Symptoms:** Low diversity in generated samples; all generated paths looking too similar; t-SNE plot showing synthetic data clustered tightly in a small region, failing to cover the entire real data manifold. Unrealistically low standard deviation across synthetic paths compared to real data.")
    st.markdown(" * **Detection:** Computing the inter-path standard deviation of synthetic data and comparing it to real historical data (e.g., `np.std(synth_close_returns.std(axis=1))` vs `np.std(real_close_sequences_for_plot.std(axis=1))`). If the synthetic value is significantly lower, it's a red flag.")
    st.markdown(" * **Mitigation:** Techniques like Wasserstein GANs (WGANs) with gradient penalty, feature matching, or training with a greater diversity of inputs can help. TimeGAN's supervised loss also inherently encourages temporal diversity.")

    st.markdown("* **Training Instability:** GANs are notoriously hard to train. The adversarial min-max game between the generator and discriminator can be unstable, leading to oscillating or non-converging loss curves (as seen in Section 3.3 for custom GANs).")
    st.markdown(" * **Symptoms:** Generator and discriminator losses fluctuating wildly or not converging to a stable equilibrium. Generated sample quality varying greatly across epochs.")
    st.markdown(" * **Mitigation:** Careful hyperparameter tuning (learning rates, batch size), using WGANs, or applying label smoothing can improve stability.")

    st.markdown("* **Hyperparameter Sensitivity:** The quality of GAN-generated data is highly sensitive to hyperparameters like learning rate, batch size, network architecture, and loss weights ($\\gamma$ in TimeGAN). A configuration that works for one dataset may not work for another.")
    st.markdown(" * **Symptoms:** Poor quality synthetic data, even after extensive training, due to suboptimal hyperparameter choices.")
    st.markdown(" * **Mitigation:** Extensive hyperparameter search, cross-validation, and potentially transfer learning from pre-trained models.")

    st.markdown("* **Non-stationarity:** A GAN trained on historical data learns the market dynamics of that specific period. Using it to generate future scenarios implicitly assumes stationarity of these dynamics, which is often violated in financial markets.")

    st.markdown("* **Validation Paradox:** If we could perfectly evaluate synthetic data quality, we wouldn't need synthetic data—we would already understand the true underlying distribution. This highlights the inherent challenge: we are using synthetic data because we don't fully understand the real data's generative process, yet we need to validate the synthetic data against that same unknown process.")

    st.markdown(
        """
Sarah understands that while TimeGAN provides a powerful tool, it requires careful implementation, monitoring, and a critical understanding of its potential pitfalls to ensure the generated data is truly valuable for financial decision-making.
        """
    )


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.write("© 2026 QuantUniversity. All Rights Reserved.")
st.caption(
    "The purpose of this demonstration is solely for educational use and illustration. "
    "To access the full legal documentation, please visit this link. Any reproduction of this demonstration "
    "requires prior written consent from QuantUniversity."
)
st.caption(
    "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, "
    "which may contain inaccuracies or errors."
)
