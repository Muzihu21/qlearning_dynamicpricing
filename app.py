import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from q_learning_env import PenjualanEnv

# ========== Setup ==========
st.set_page_config(page_title="Q-Learning Harga", layout="wide")

# ========== Load Environment ==========
env = PenjualanEnv()
env.unique_states = list(set(env.states))
env.n_states = len(env.unique_states)

# ========== Sidebar Menu ==========
menu = st.sidebar.radio("Pilih Halaman", [
    "📊 Visualisasi Q-table",
    "📈 Evaluasi Policy",
    "📉 Grafik Reward",
    "⚙️ Training Ulang",
    "📋 Peta Harga Produk",
    "ℹ️ Tentang"
])

# ========== Fungsi: Training ==========
def train_q_learning(env, alpha, gamma, epsilon, episodes):
    state_to_index = {s: i for i, s in enumerate(env.unique_states)}
    q_table = np.zeros((len(env.unique_states), env.n_actions))
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_idx = state_to_index[state]
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[state_idx])

            result = env.step(action)
            next_state, reward, done = result[:3]
            next_state_idx = state_to_index[next_state]
            q_table[state_idx, action] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
            )

            total_reward += reward
            state = next_state
        rewards_per_episode.append(total_reward)

    return q_table, np.array(rewards_per_episode)

# ========== Fungsi: Evaluasi ==========
def evaluate_policy(env, q_table, n_trials=100):
    total_rewards = []
    harga_history = []

    for _ in range(n_trials):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            try:
                state_index = min(env.unique_states.index(state), q_table.shape[0] - 1)
                action = np.argmax(q_table[state_index])
                result = env.step(action)
                next_state, reward, done = result[:3]
                harga_idx = next_state[0]
                harga = env.harga_list[harga_idx]
                harga_history.append(harga)

                episode_reward += reward
                state = next_state
            except Exception as e:
                st.warning(f"⚠️ Error evaluasi policy: {e}")
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), harga_history

# ========== Halaman: Visualisasi Q-table ==========
if menu == "📊 Visualisasi Q-table":
    st.title("📊 Strategi Harga: Q-table Heatmap")
    
    try:
        if "q_table" in st.session_state and st.session_state.get("just_trained", False):
            q_table = st.session_state["q_table"]
            st.info("Menampilkan Q-table hasil training terbaru.")
        else:
            q_table = np.load("q_table.npy").astype(float)
            
        xticklabels = ["Turun", "Tetap", "Naik"]
        yticklabels = [f"H{int(state[0])}-P{int(state[1])}" for state in env.unique_states]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(q_table, annot=True, fmt=".0f", cmap="YlGnBu",
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    linewidths=0.5, linecolor='gray', ax=ax)
        ax.set_xlabel("Harga (Action)")
        ax.set_ylabel("State")
        ax.set_title("Q-Table Heatmap")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state["just_trained"] = False

    except FileNotFoundError:
        st.error("❌ File `q_table.npy` tidak ditemukan. Silakan lakukan training terlebih dahulu.")
    except Exception as e:
        st.error(f"❌ Gagal memuat atau menampilkan Q-table: {e}")

# ========== Halaman: Evaluasi Policy ==========
elif menu == "📈 Evaluasi Policy":
    st.title("📈 Evaluasi Policy")
    try:
        q_table = np.load("q_table.npy")
        trials = st.slider("Jumlah Simulasi Episode", 10, 10000, 100, step=100)
        avg
