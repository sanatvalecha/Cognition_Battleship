# Cognition Battleship

A lightweight Battleship game built with Streamlit and a simple AI opponent, optimized for desktop usage. Click on the AI board to fire, watch hits and misses update instantly, and review a clean, correctly-numbered move log. Includes optional pre-game ship repositioning, a randomize button, and minimal configuration. The game is not designed for mobile use, but is usable in landscape orientation. 

## Features
- Clickable AI board; immediate UI feedback
- Clean visual markers
  - 💣 = hit
  - 💧 = miss
  - 💥 = tiles of a sunk ship
  - 🚢 = your ships (visible only on your board)
- Move log with true sequential numbering (newest on top)
- Randomize My Ships (before first move)
- Optional pre-game ship repositioning (pick up/rotate/drop; disabled after first move)
- Vertical divider between boards
- OpenAI integration via `st.secrets` (no inputs in UI)

## Project structure
```
Cognition_Battleship/
├─ app.py                 # Streamlit app (UI, game logic, AI integration)
├─ prompts/
│  └─ prompt_ai_move.txt  # Prompt template for AI targeting (optional override)
├─ .streamlit/
│  └─ secrets.toml        # Stores OPENAI_API_KEY
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```

## Prerequisites
- Python 3.9+
- An OpenAI API key stored in Streamlit secrets

Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# .venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Then open the printed Local URL.

## Sidebar controls
- New Game / Reset: start a fresh game
- Randomize My Ships: re-roll your ship placement (only before first move)
- Reposition Ships (pre-game only):
  - Start: enter reposition mode
  - Orientation: toggle H/V while holding a ship
  - Finish: exit reposition mode

## How to play
1. Optionally randomize or reposition your ships before the first move.
2. Click a cell on the AI board to fire.
3. The AI replies automatically; your board updates instantly.
4. Continue until all ships of one side are sunk.

## AI
- Uses OpenAI Chat Completions if an API key is present in `st.secrets`.
- Falls back to heuristics plus random selection when needed.
- You can customize `prompts/prompt_ai_move.txt` to adjust behavior.

## Troubleshooting
- Missing API key: ensure `.streamlit/secrets.toml` has a valid `OPENAI_API_KEY`.
- Streamlit not installed: `pip install -r requirements.txt`.
- Port in use: `streamlit run app.py --server.port 8502`.
- Emoji visibility: app includes CSS to center/clarify emojis in light/dark modes.

## License
MIT
