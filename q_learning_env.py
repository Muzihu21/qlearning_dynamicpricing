import pandas as pd
import random
from itertools import product  # Penting biar semua state kebentuk

class PenjualanEnv:
    def __init__(self, data_path='env_ready_data.csv', max_steps=10):
        self.data = pd.read_csv(data_path)

        # Harga dan penjualan unik
        self.unique_harga = sorted(self.data['harga_index'].unique())
        self.unique_penjualan = sorted(self.data['penjualan_level'].unique())

        # Kombinasi semua state
        self.states = list(product(self.unique_harga, self.unique_penjualan))
        self.unique_states = self.states

        self.n_actions = 3  # 0: Turun, 1: Tetap, 2: Naik

        # Reward dictionary
        self.reward_dict = {
            (row['harga_index'], row['penjualan_level']): row['reward']
            for _, row in self.data.iterrows()
        }

        # Harga asli
        if 'harga' in self.data.columns:
            harga_df = self.data[['harga_index', 'harga']].drop_duplicates().sort_values(by='harga_index')
            self.harga_list = harga_df['harga'].tolist()
        else:
            self.harga_list = self.unique_harga

        # 🔧 Mapping harga_index → posisi 0-based
        self.harga_idx_to_pos = {v: i for i, v in enumerate(self.unique_harga)}
        self.pos_to_harga_idx = {i: v for i, v in enumerate(self.unique_harga)}  # optional

        # Mapping state ke index
        self.state_to_index = {s: i for i, s in enumerate(self.unique_states)}
        self.index_to_state = {i: s for s, i in self.state_to_index.items()}

        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.current_step = 0
        self.state = random.choice(self.unique_states)
        return self.state

    def step(self, action):
        harga_idx, penjualan_lvl = self.state
        pos = self.harga_idx_to_pos[harga_idx]

        # Aksi
        if action == 0 and pos > 0:
            pos -= 1
        elif action == 2 and pos < len(self.unique_harga) - 1:
            pos += 1
        # aksi 1: tetap

        # Dapatkan harga_idx baru dari pos baru
        harga_idx = self.unique_harga[pos]

        # Pilih next state acak dari level penjualan
        possible_states = [(harga_idx, p) for p in self.unique_penjualan]
        next_state = random.choice(possible_states)
        self.state = next_state

        self.current_step += 1
        reward = self.reward_dict.get(self.state, 0)

        done = self.current_step >= self.max_steps
        return self.state, reward, done
