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

if menu == "📊 Visualisasi Q-table":
    st.title("📊 Strategi Harga: Tabel Keputusan Sederhana")

    st.markdown("""
    Berikut adalah rekomendasi aksi terbaik untuk setiap kombinasi 
    **harga saat ini** dan **permintaan pasar**.
    Semua nilai sudah didasarkan pada hasil training menggunakan Q-Learning.
    """)

    try:
        if "q_table" in st.session_state and st.session_state.get("just_trained", False):
            q_table = st.session_state["q_table"]
            st.info("Menampilkan Q-table hasil training terbaru.")
        else:
            q_table = np.load("q_table.npy").astype(float)

        df_decision = visualize_simple_decision_table(q_table, env)
        st.dataframe(df_decision, use_container_width=True)

    except FileNotFoundError:
        st.error("❌ File `q_table.npy` tidak ditemukan. Silakan lakukan training terlebih dahulu.")
    except Exception as e:
        st.error(f"❌ Gagal memuat atau menampilkan tabel keputusan: {e}")

# ========== Halaman: Visualisasi Q-table ==========
if menu == "📊 Visualisasi Q-table":
    st.title("📊 Strategi Harga: Q-table Heatmap")

    st.markdown("""
    **Cara Membaca Heatmap:**
    - **Sumbu Y (Kondisi)**: Misalnya, `H3-P1` artinya harga ke-4 dengan permintaan sedang.
    - **Sumbu X (Aksi)**: `Turun`, `Tetap`, atau `Naik`.
    - **Warna**: Semakin terang warna, semakin tinggi nilai Q → tindakan semakin menguntungkan.
    """)

    try:
        if "q_table" in st.session_state and st.session_state.get("just_trained", False):
            q_table = st.session_state["q_table"]
            st.info("Menampilkan Q-table hasil training terbaru.")
        else:
            q_table = np.load("q_table.npy").astype(float)

        visualize_q_table_simple(q_table, env)

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
        avg_reward, harga_history = evaluate_policy(env, q_table, trials)
        st.success(f"🎯 Rata-rata reward dari {trials} simulasi: **{avg_reward:.2f}**")

        st.markdown("### 📊 Perubahan Harga Selama Evaluasi")
        fig, ax = plt.subplots()
        ax.plot(harga_history, color='orange')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Harga Produk")
        ax.set_title("Perubahan Harga Selama Evaluasi")
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("❌ File `q_table.npy` tidak ditemukan.")

# ========== Halaman: Grafik Reward ==========
elif menu == "📉 Grafik Reward":
    st.title("📉 Grafik Reward per Episode")

    if "rewards" in st.session_state and st.session_state.get("just_trained", False):
        rewards = st.session_state["rewards"]
        st.info("Menampilkan hasil training terbaru.")
    else:
        try:
            rewards = np.load("rewards_per_episode.npy")
            st.caption("Data dimuat dari file rewards_per_episode.npy")
        except FileNotFoundError:
            st.error("❌ File `rewards_per_episode.npy` tidak ditemukan.")
            rewards = None

    if rewards is not None:
        fig, ax = plt.subplots()
        ax.plot(rewards, label='Reward per Episode', color='green')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward per Episode (Training Progress)")
        ax.legend()
        st.pyplot(fig)

        total_profit = np.sum(rewards)
        avg_profit = np.mean(rewards)

        col1, col2 = st.columns(2)
        col1.metric("📈 Total Profit", f"Rp {total_profit:,.0f}")
        col2.metric("💰 Rata-rata Profit/Episode", f"Rp {avg_profit:,.0f}")

        st.session_state["just_trained"] = False

# ========== Halaman: Training Ulang ==========
elif menu == "⚙️ Training Ulang":
    st.title("⚙️ Training Ulang Q-Learning")

    alpha = st.number_input("Alpha (Learning rate)", 0.0, 1.0, 0.1, step=0.01)
    gamma = st.number_input("Gamma (Discount factor)", 0.0, 1.0, 0.9, step=0.01)
    epsilon = st.number_input("Epsilon (Exploration rate)", 0.0, 1.0, 0.1, step=0.01)
    episodes = st.number_input("Jumlah Episode", 100, 10000, 1000, step=100)

    if st.button("🚀 Mulai Training"):
        with st.spinner("Training sedang berjalan..."):
            q_table, rewards = train_q_learning(env, alpha, gamma, epsilon, episodes)
            np.save("q_table.npy", q_table)
            np.save("rewards_per_episode.npy", rewards)
            st.session_state["rewards"] = rewards
            st.session_state["q_table"] = q_table
            st.session_state["just_trained"] = True
            st.success("✅ Training selesai dan file disimpan. Silakan cek grafik reward.")

# ========== Halaman: Peta Harga Produk ==========
elif menu == "📋 Peta Harga Produk":
    st.title("📋 Rekomendasi Harga Produk Setelah Training")

    try:
        df_produk = pd.read_csv("produk.csv")
        q_table = np.load("q_table.npy")
        best_actions = np.argmax(q_table, axis=1)

        rekomendasi_per_harga = {}
        for harga_awal in env.harga_list:
            matching_states = [i for i, state in enumerate(env.unique_states) if env.harga_list[state[0]] == harga_awal]
            if not matching_states:
                aksi_terbaik = 1
            else:
                actions = [best_actions[i] for i in matching_states]
                aksi_terbaik = max(set(actions), key=actions.count)

            harga_idx = env.harga_list.index(harga_awal)
            if aksi_terbaik == 0 and harga_idx > 0:
                harga_idx -= 1
            elif aksi_terbaik == 2 and harga_idx < len(env.harga_list) - 1:
                harga_idx += 1

            harga_final = env.harga_list[harga_idx]
            rekomendasi_per_harga[harga_awal] = harga_final

        df_produk["Harga Awal"] = df_produk["Harga (Rp)"]
        df_produk["Rekomendasi Harga"] = df_produk["Harga Awal"].apply(
            lambda h: rekomendasi_per_harga.get(h, h)
        )

        missing = df_produk[~df_produk["Harga Awal"].isin(rekomendasi_per_harga.keys())]
        if not missing.empty:
            st.warning(f"⚠️ {len(missing)} produk tidak memiliki rekomendasi eksplisit. "
                       "Harga awal digunakan sebagai fallback.")

        df_produk = df_produk[["id_produk", "Nama Produk", "Kategori", "Harga Awal", "Rekomendasi Harga"]]
        st.dataframe(df_produk)

    except Exception as e:
        st.error(f"❌ Gagal menampilkan harga: {e}")

# ========== Halaman: Tentang ==========
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai bagian dari skripsi untuk mensimulasikan **Reinforcement Learning (Q-Learning)** 
    dalam konteks **penetapan harga produk**.

    **Fitur:**
    - Visualisasi Q-table (heatmap)
    - Evaluasi policy dan pergerakan harga
    - Grafik reward per episode
    - Training ulang dengan hyperparameter custom
    - Rekomendasi harga per produk berdasarkan Q-table

    **Author**: Zihu — AI Engineer & Pejuang Skripsi 🧠🔥  
    **Stack**: Python, Streamlit, NumPy, Matplotlib, Seaborn
    """)

# ========== Footer ==========
st.markdown("---")
st.caption("© 2025 — Made with ❤️ by Zihu | Powered by Streamlit")
