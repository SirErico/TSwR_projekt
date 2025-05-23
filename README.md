# 🤖 TSwR_projekt

**Porównanie uczenia ze wzmocnieniem (Reinforcement Learning) i klasycznego sterowania z linearyzacją przez sprzężenie zwrotne**

Projekt wykorzystuje środowisko [Reacher (Mujoco)](https://gymnasium.farama.org/environments/mujoco/reacher/) z biblioteki Gymnasium.

<p align="center">
  <img src="https://gymnasium.farama.org/_images/reacher.gif" alt="mujoco reacher" width="400">
</p>

- [More about the MuJoCo Reacher environment](reacher_info.md)
- [RL algorithms we used](rl/rl_algorithms.md)
---

## 🎯 Cele projektu

1. **Uczenie ze wzmocnieniem (RL)**  
   Trening agenta do jak najszybszego osiągnięcia celu w środowisku Reacher.

2. **Sterowanie klasyczne**  
   Identyfikacja modelu oraz implementacja sterowania z linearyzacją przez sprzężenie zwrotne (feedback linearization) i planowania ruchu.

3. **Porównanie metod**  
   Analiza wydajności obu podejść pod względem dokładności, stabilności i szybkości osiągnięcia celu.

---

## 🛠️ Instalacja i uruchomienie

### Klonowanie repozytorium
```bash
git clone https://github.com/SirErico/TSwR_projekt
cd TSwR_projekt
```

### Tworzenie środowiska virtualenv
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```