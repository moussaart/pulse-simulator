# PULSE Online AI Training Client

This folder contains the **external client files** needed to train a Reinforcement Learning agent (using OpenAI Gym / Gymnasium and PyTorch) that interacts with the PULSE Simulator.

You can copy this folder **anywhere** on your computer. It does not need to be inside the PULSE project!

## How it Works

We use a Client-Server architecture to make sure your AI code doesn't interfere with the complex physics and GUI of the simulator:

1. **The Server (Inside PULSE Simulator)**: 
   We added a custom algorithm called `AI Gym Server (Port 5555)` directly into the simulator's code (`src/user_algorithms/ai_socket_server.py`). 
   When you select this algorithm in the GUI, it starts listening for an AI Agent to connect.

2. **The Client (This folder)**:
   The `pulse_rl_env.py` file creates a standard `gymnasium.Env`. It connects to the GUI via a TCP socket.
   It receives the current measurements (state) and true positions (for rewards), sends back the chosen anchors (action), and lets the GUI do the heavy mathematically filtering (EKF) using **only** the anchors the AI selected!

## Setup

1. Make sure you have the basics installed for your AI environment:
   ```bash
   pip install gymnasium numpy
   # And PyTorch / stable-baselines3 if you want to use the standard RL loops
   ```

2. Open the PULSE Simulator GUI.
3. Go to the **Algorithms** selection panel.
4. Choose **AI Gym Server (Port 5555)** from the drop-down list.
5. Click **Start** to run the simulation scenario. The simulation will pause or run slowly while waiting for the AI to connect.

## Training

Open a new terminal (in this external folder) and run the example script:

```bash
python train_agent.py
```

Watch the terminal and the GUI. As the AI Agent plays, you will see the GUI update dynamically based on the anchors the AI decides to use. The AI is rewarded for selecting Line-of-Sight (LOS) anchors and penalized for using Non-Line-of-Sight (NLOS) anchors.

## Next Steps

Edit `train_agent.py` to replace the "dummy" random agent with a real PyTorch model using your preferred library (like Stable-Baselines3's PPO or SAC).
