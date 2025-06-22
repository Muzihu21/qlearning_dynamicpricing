import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from q_learning_env import PenjualanEnv

# ===================== Setup =====================
st.set_page_config(page_title="Simulasi Q-Learning", layout="wide")

st.title("📈 Simulasi Q-Learning untuk Penetapan Harga Produk")
st.caption("Studi Kasus: Perusahaan Digzi")
st.markdown("---")

# ===================== Load Environment =====================
import os

data_path = "env_ready_data.csv"
if not os.path.exists(data_path):
    st.error(f"❌ File {data_path} tidak ditemukan. Pastikan file tersedia di direktori yang sama dengan app.py")
    st.stop()

env = PenjualanEnv(data_path=data_path)
env.unique_states = list(set(env.states))
env.n_states = len(env.unique_states)

# ===================== Sidebar =====================
menu = st.sidebar.radio("📋 Navigasi", [
    "🗂️ Data Awal & Persiapan",
    "🏠 Beranda",
    "📊 Q-Table Heatmap",
    "📉 Grafik Reward",
    "🧪 Evaluasi Policy",
    "⚙️ Training Ulang"
])

# ===================== Fungsi Training =====================
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

            next_state, reward, done = env.step(action)
            next_state_idx = state_to_index[next_state]

            q_table[state_idx, action] += alpha * (
                reward + gamma * np.max(q_table[next_state_idx]) - q_table[state_idx, action]
            )

            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)

    return q_table, np.array(rewards_per_episode)

# ===================== Fungsi Evaluasi =====================
def evaluate_policy(env, q_table, n_trials=100):
    total_rewards = []
    for _ in range(n_trials):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            try:
                state_index = env.unique_states.index(state)
                action = np.argmax(q_table[state_index])
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
            except Exception as e:
                st.warning(f"⚠️ Error evaluasi policy: {e}")
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# ===================== Halaman: Data Awal & Persiapan =====================
if menu == "🗂️ Data Awal & Persiapan":
    st.subheader("📦 Data Penjualan (env_ready_data.csv)")
    st.markdown("""
    Data ini berisi histori kombinasi harga dan tingkat penjualan dari perusahaan Digzi.
    Setiap baris merepresentasikan satu kondisi pasar, termasuk reward (nilai ekonomi) dari strategi tersebut.
    """)
    try:
        st.dataframe(env.data.head(20))
        st.markdown(f"Total baris data: **{len(env.data):,}**".replace(",", "."))

        st.markdown("---")
        st.subheader("📊 Distribusi Harga dan Penjualan")
        fig1, ax1 = plt.subplots()
        sns.countplot(x=env.data['harga_index'], ax=ax1, palette='Blues')
        ax1.set_title("Frekuensi Harga Index")
        ax1.set_xlabel("Harga Index")
        ax1.set_ylabel("Jumlah")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        sns.countplot(x=env.data['penjualan_level'], ax=ax2, palette='Greens')
        ax2.set_title("Frekuensi Level Penjualan")
        ax2.set_xlabel("Level Penjualan")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.histplot(env.data['reward'], bins=30, kde=True, color='orange', ax=ax3)
        ax3.set_title("Distribusi Reward")
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Frekuensi")
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"Gagal memuat data: {e}")

# ===================== Halaman: Beranda =====================
if menu == "🏠 Beranda":
    st.subheader("Selamat datang di Aplikasi Simulasi Q-Learning")
    st.markdown("""
    Aplikasi ini mensimulasikan algoritma **Q-Learning** untuk mengoptimalisasi strategi harga produk. 
    Dataset digunakan dari data penjualan perusahaan **Digzi**.

    Navigasi di sebelah kiri dapat digunakan untuk:
    - Melihat data awal dan distribusinya
    - Melihat Q-table hasil training
    - Melihat performa strategi
    - Melatih ulang model dengan hyperparameter yang bisa diatur
    """)

# ===================== Halaman: Heatmap =====================
elif menu == "📊 Q-Table Heatmap":
    st.subheader("📊 Visualisasi Q-Table")
    try:
        q_table = np.load("q_table.npy")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(q_table, annot=True, fmt=",.0f", cmap="YlOrRd",
                    xticklabels=env.harga_list,
                    yticklabels=env.unique_states,
                    cbar_kws={'label': 'Nilai Q'})
        ax.set_xlabel("Aksi Harga")
        ax.set_ylabel("State")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Gagal memuat Q-table: {e}")

elif menu == "📉 Grafik Reward":
    st.subheader("📉 Reward Selama Training")
    try:
        rewards = np.load("rewards_per_episode.npy")
        total_episodes = len(rewards)
        total_reward = int(np.sum(rewards))
        avg_reward = int(np.mean(rewards))
        max_reward = int(np.max(rewards))
        min_reward = int(np.min(rewards))

        # Helper buat format angka
        def format_id(x): 
            return f"{x:,}".replace(",", ".")

        # Tampilan Metric
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Rata-rata Reward", format_id(avg_reward))
        col2.metric("💰 Total Reward", format_id(total_reward))
        col3.metric("📈 Maksimum", format_id(max_reward))
        col4.metric("📉 Minimum", format_id(min_reward))

        # 1️⃣ Reward per Episode
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(rewards, color='green')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Reward per Episode")
        st.pyplot(fig1)

        # 2️⃣ Distribusi Reward
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.histplot(rewards, kde=True, color='skyblue', ax=ax2)
        ax2.set_title("Distribusi Reward")
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Frekuensi")
        st.pyplot(fig2)
    except FileNotFoundError:
        st.error("❌ File reward tidak ditemukan.")

    except FileNotFoundError:
        st.error("❌ File reward tidak ditemukan.")


# ===================== Halaman: Evaluasi =====================
elif menu == "🧪 Evaluasi Policy":
    st.subheader("🧪 Evaluasi Strategi Hasil Q-Learning")
    with st.form("eval_form"):
        col1, col2, col3, col4, col5 = st.columns(5)
        alpha = col1.number_input("Alpha", 0.0, 1.0, 0.1)
        gamma = col2.number_input("Gamma", 0.0, 1.0, 0.9)
        epsilon = col3.number_input("Epsilon", 0.0, 1.0, 0.1)
        episodes = col4.number_input("Episodes", 100, 10000, 1000, step=100)
        trials = col5.number_input("Simulasi Evaluasi", 10, 10000, 100, step=100)
        eval_btn = st.form_submit_button("🚀 Latih & Evaluasi")

    if eval_btn:
        with st.spinner("Sedang melatih dan mengevaluasi..."):
            q_table, rewards = train_q_learning(env, alpha, gamma, epsilon, episodes)
            avg_reward = evaluate_policy(env, q_table, trials)

            def format_id(x):
                return f"{x:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
            trials_formatted = f"{trials:,}".replace(",", ".")

            st.success(f"🎯 Rata-rata reward dari {trials_formatted} simulasi: **{format_id(avg_reward)}**")

            st.markdown("---")
            st.subheader("📉 Reward Hasil Training")
            total_reward = int(np.sum(rewards))
            max_reward = int(np.max(rewards))
            min_reward = int(np.min(rewards))
            avg_episode_reward = int(np.mean(rewards))

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Rata-rata Reward/Episode", format_id(avg_episode_reward))
            col2.metric("💰 Total Reward", format_id(total_reward))
            col3.metric("📈 Maksimum", format_id(max_reward))
            col4.metric("📉 Minimum", format_id(min_reward))

            fig1, ax1 = plt.subplots()
            ax1.plot(rewards, color='royalblue')
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Reward per Episode (Evaluasi)")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            sns.histplot(rewards, kde=True, color='lightgreen', ax=ax2)
            ax2.set_title("Distribusi Reward Evaluasi")
            ax2.set_xlabel("Reward")
            ax2.set_ylabel("Frekuensi")
            st.pyplot(fig2)

            if len(rewards) >= 100:
                rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
                fig3, ax3 = plt.subplots()
                ax3.plot(rolling_avg, color='orange')
                ax3.set_title("Rolling Average (100 episode)")
                ax3.set_xlabel("Episode")
                ax3.set_ylabel("Rata-rata Reward")
                st.pyplot(fig3)

# ===================== Halaman: Training =====================
elif menu == "⚙️ Training Ulang":
    st.subheader("⚙️ Training Ulang Model Q-Learning")
    with st.form("training_form"):
        col1, col2, col3, col4 = st.columns(4)
        alpha = col1.number_input("Alpha", 0.0, 1.0, 0.1)
        gamma = col2.number_input("Gamma", 0.0, 1.0, 0.9)
        epsilon = col3.number_input("Epsilon", 0.0, 1.0, 0.1)
        episodes = col4.number_input("Episodes", 100, 10000, 1000, step=100)
        submitted = st.form_submit_button("🚀 Mulai Training")

        if submitted:
            with st.spinner("Training sedang berjalan..."):
                q_table, rewards = train_q_learning(env, alpha, gamma, epsilon, episodes)
                np.save("q_table.npy", q_table)
                np.save("rewards_per_episode.npy", rewards)
                st.success("✅ Training selesai dan file disimpan.")

# ===================== Footer =====================
st.markdown("---")
st.caption("© 2025 | Dibuat oleh MuZihu untuk skripsi Digzi | Powered by Streamlit")
