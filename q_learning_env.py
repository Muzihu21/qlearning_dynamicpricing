import pandas as pd
import random
from itertools import product  # Penting biar semua state kebentuk

class PenjualanEnv:
    def __init__(self, data_path='env_ready_data.csv', max_steps=10):
        self.data = pd.read_csv(data_path)

        # Ambil harga dan penjualan unik
        self.unique_harga = sorted(self.data['harga_index'].unique())
        self.unique_penjualan = sorted(self.data['penjualan_level'].unique())

        # Buat semua kombinasi state (harga_index x penjualan_level)
        self.states = list(product(self.unique_harga, self.unique_penjualan))
        self.unique_states = self.states

        # Jumlah aksi
        self.n_actions = 3  # 0: Turun, 1: Tetap, 2: Naik

        # Bikin dict reward supaya lookup cepat
        self.reward_dict = {
            (row['harga_index'], row['penjualan_level']): row['reward']
            for _, row in self.data.iterrows()
        }

        # Harga asli sesuai harga_index
        if 'harga' in self.data.columns:
            harga_df = self.data[['harga_index', 'harga']].drop_duplicates().sort_values(by='harga_index')
            self.harga_list = harga_df['harga'].tolist()
        else:
            self.harga_list = self.unique_harga

        # Mapping state ke index untuk q_table
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

        # Update harga sesuai aksi
        if action == 0 and harga_idx > min(self.unique_harga):
            harga_idx -= 1
        elif action == 2 and harga_idx < max(self.unique_harga):
            harga_idx += 1
        # aksi 1 = tetap (gak berubah)

        # Pilih state baru dengan harga_idx yang sama, tapi random penjualan_level
        possible_states = [(harga_idx, p) for p in self.unique_penjualan]
        next_state = random.choice(possible_states)
        self.state = next_state

        self.current_step += 1

        # Ambil reward, fallback 0 kalau gak ada
        reward = self.reward_dict.get(self.state, 0)

        done = self.current_step >= self.max_steps
        return self.state, reward, done
