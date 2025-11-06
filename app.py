import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Library missing until requirements are installed

# --- Constants ---
BOARD_SIZE = 10
LETTERS = "ABCDEFGHIJ"
DEFAULT_MODEL = "gpt-4o-mini"
SHIP_TYPES = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}

# --- Helpers ---
def coords_to_label(r: int, c: int) -> str:
    return f"{LETTERS[r]}{c+1}"

def label_to_coords(label: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"^\s*([A-Ja-j])\s*(10|[1-9])\s*$", label)
    if not m:
        return None
    r = LETTERS.index(m.group(1).upper())
    c = int(m.group(2)) - 1
    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
        return (r, c)
    return None

def empty_grid(val=None):
    return [[val for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def place_ships_randomly() -> Tuple[List[List[str]], List[List[Optional[str]]], Dict[str, int]]:
    grid = empty_grid("")  # "", "H", "M", "S"
    ship_map = empty_grid(None)  # ship name or None
    health = {name: size for name, size in SHIP_TYPES.items()}

    for name, size in SHIP_TYPES.items():
        placed = False
        attempts = 0
        while not placed and attempts < 1000:
            attempts += 1
            horizontal = bool(random.getrandbits(1))
            if horizontal:
                r = random.randint(0, BOARD_SIZE - 1)
                c = random.randint(0, BOARD_SIZE - size)
                if all(ship_map[r][c + k] is None for k in range(size)):
                    for k in range(size):
                        ship_map[r][c + k] = name
                        grid[r][c + k] = "S"
                    placed = True
            else:
                r = random.randint(0, BOARD_SIZE - size)
                c = random.randint(0, BOARD_SIZE - 1)
                if all(ship_map[r + k][c] is None for k in range(size)):
                    for k in range(size):
                        ship_map[r + k][c] = name
                        grid[r + k][c] = "S"
                    placed = True
        if not placed:
            raise RuntimeError("Failed to place ships")
    return grid, ship_map, health

def can_place_ship(ship_map: List[List[Optional[str]]], name: str, size: int, r: int, c: int, horizontal: bool) -> bool:
    if horizontal:
        if c + size > BOARD_SIZE:
            return False
        return all(ship_map[r][c + k] is None for k in range(size))
    else:
        if r + size > BOARD_SIZE:
            return False
        return all(ship_map[r + k][c] is None for k in range(size))

def place_ship_on_board(grid: List[List[str]], ship_map: List[List[Optional[str]]], name: str, size: int, r: int, c: int, horizontal: bool) -> None:
    if horizontal:
        for k in range(size):
            ship_map[r][c + k] = name
            grid[r][c + k] = "S"
    else:
        for k in range(size):
            ship_map[r + k][c] = name
            grid[r + k][c] = "S"

def get_ship_cells(ship_map: List[List[Optional[str]]], r: int, c: int) -> Tuple[Optional[str], List[Tuple[int,int]], str]:
    name = ship_map[r][c]
    if not name:
        return None, [], "H"
    # Try horizontal span
    cells = []
    cc = c
    while cc >= 0 and ship_map[r][cc] == name:
        cc -= 1
    cc += 1
    while cc < BOARD_SIZE and ship_map[r][cc] == name:
        cells.append((r, cc))
        cc += 1
    if len(cells) > 1:
        return name, cells, "H"
    # Try vertical span
    cells = []
    rr = r
    while rr >= 0 and ship_map[rr][c] == name:
        rr -= 1
    rr += 1
    while rr < BOARD_SIZE and ship_map[rr][c] == name:
        cells.append((rr, c))
        rr += 1
    return name, cells, "V"

def perform_attack(grid: List[List[str]], ship_map: List[List[Optional[str]]], ships_health: Dict[str, int], r: int, c: int) -> Tuple[str, Optional[str]]:
    # returns ("miss"|"hit"|"sunk"|"repeat", sunk_ship_name_if_any)
    if grid[r][c] in ("H", "M"):
        return "repeat", None
    ship = ship_map[r][c]
    if ship:
        grid[r][c] = "H"
        ships_health[ship] -= 1
        if ships_health[ship] == 0:
            return "sunk", ship
        return "hit", None
    else:
        grid[r][c] = "M"
        return "miss", None

def all_sunk(ships_health: Dict[str, int]) -> bool:
    return all(v == 0 for v in ships_health.values())

def load_prompt_text(board_size: int, history_json: str, already_tried: List[str]) -> str:
    prompts_path = Path(__file__).parent / "prompts" / "prompt_ai_move.txt"
    base = (
        "You are the AI player in a standard 10x10 Battleship game.\n"
        "Board size: {board_size}x{board_size}\n"
        "History: {history_json}\n"
        "Tried: {already_tried}\n"
        "Output ONLY one coordinate like B7 or J10."
    )
    if prompts_path.exists():
        template = prompts_path.read_text(encoding="utf-8")
    else:
        template = base
    try:
        return template.format(
            board_size=board_size,
            history_json=history_json,
            already_tried=", ".join(already_tried),
        )
    except Exception:
        # Fallback to simple formatting if template has unexpected braces
        return base.format(
            board_size=board_size,
            history_json=history_json,
            already_tried=", ".join(already_tried),
        )

def get_openai_client(api_key: Optional[str]) -> Optional["OpenAI"]:
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def ai_pick_coordinate(
    api_key: Optional[str],
    model: str,
    ai_history: List[Dict[str, str]],
    already_tried: List[str],
) -> Optional[Tuple[int, int]]:
    tried_set = set(already_tried)

    def in_bounds(rr: int, cc: int) -> bool:
        return 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE

    def neighbors_of(r: int, c: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if in_bounds(rr, cc):
                yield rr, cc

    # Prioritize targeting adjacent to hits since the last sunk
    last_sunk_idx = -1
    for i in range(len(ai_history) - 1, -1, -1):
        if ai_history[i].get("result") == "sunk":
            last_sunk_idx = i
            break

    pending_hits: List[Tuple[int, int]] = []
    for j in range(last_sunk_idx + 1, len(ai_history)):
        it = ai_history[j]
        if it.get("result") == "hit":
            t = label_to_coords(it.get("coord", ""))
            if t:
                pending_hits.append(t)

    if pending_hits:
        candidates: List[Tuple[int, int]] = []
        if len(pending_hits) >= 2:
            rows = [r for r, _ in pending_hits]
            cols = [c for _, c in pending_hits]
            if len(set(rows)) == 1:
                row = rows[0]
                min_c, max_c = min(cols), max(cols)
                for cc in (min_c - 1, max_c + 1):
                    if in_bounds(row, cc):
                        lbl = coords_to_label(row, cc)
                        if lbl not in tried_set:
                            candidates.append((row, cc))
            elif len(set(cols)) == 1:
                col = cols[0]
                min_r, max_r = min(rows), max(rows)
                for rr in (min_r - 1, max_r + 1):
                    if in_bounds(rr, col):
                        lbl = coords_to_label(rr, col)
                        if lbl not in tried_set:
                            candidates.append((rr, col))
        if not candidates:
            for (r, c) in pending_hits:
                for rr, cc in neighbors_of(r, c):
                    lbl = coords_to_label(rr, cc)
                    if lbl not in tried_set:
                        candidates.append((rr, cc))
        if candidates:
            return random.choice(candidates)

    # Prepare prompt
    history_json = json.dumps(ai_history, ensure_ascii=False)
    prompt_text = load_prompt_text(BOARD_SIZE, history_json, already_tried)

    # Call OpenAI if available
    client = get_openai_client(api_key)
    candidate: Optional[Tuple[int, int]] = None
    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt_text}],
                temperature=0.2,
                max_tokens=6,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\b([A-Ja-j])\s*(10|[1-9])\b", text)
            if m:
                label = f"{m.group(1).upper()}{m.group(2)}"
                if label not in tried_set:
                    candidate = label_to_coords(label)
        except Exception as e:
            candidate = None

    # Fallback: checkerboard, then any remaining
    if candidate is None:
        all_coords = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
        checker = [(r, c) for (r, c) in all_coords if ((r + c) % 2 == 0) and coords_to_label(r, c) not in tried_set]
        if checker:
            candidate = random.choice(checker)
        else:
            remaining = [(r, c) for (r, c) in all_coords if coords_to_label(r, c) not in tried_set]
            if not remaining:
                return None
            candidate = random.choice(remaining)
    return candidate

# --- Game initialization ---
def new_game_state():
    player_grid, player_ship_map, player_health = place_ships_randomly()
    ai_grid, ai_ship_map, ai_health = place_ships_randomly()

    return {
        "turn": "player",  # "player" or "ai"
        "game_over": False,
        "winner": None,
        "log": [],  # legacy log (not used for numbering)
        "moves": [],  # numbered move log, each entry is one shot
        "move_no": 0,  # increments once per shot (player or AI)
        # Boards
        "player_grid": player_grid,
        "player_ship_map": player_ship_map,
        "player_health": player_health,
        "ai_grid": ai_grid,
        "ai_ship_map": ai_ship_map,
        "ai_health": ai_health,
        # Histories
        "player_attacks": set(),  # {(r,c)}
        "ai_attacks": set(),
        "ai_history": [],  # [{coord, result}]
        # Control
        "ai_should_move": False,
        # Reposition state (pre-game only)
        "reposition": False,
        "reposition_pick": None,  # {name, size}
        "reposition_orientation": "H",
    }

# --- Rendering ---
def render_player_board(game, on_click_cell=None):
    st.subheader("Your Board")
    for r in range(BOARD_SIZE):
        cols = st.columns(BOARD_SIZE, gap="small")
        for c in range(BOARD_SIZE):
            cell = game["player_grid"][r][c]
            # Show ships; use emojis
            ship_name = game["player_ship_map"][r][c]
            if ship_name and game["player_health"][ship_name] == 0:
                label = "💥"  # entire sunk ship shows explosion
            elif cell == "H":
                label = "💣"  # regular hit shows bomb
            elif cell == "M":
                label = "💧"
            elif cell == "S":
                label = "🚢"
            else:
                label = " "
            can_click = game.get("reposition", False) and not game.get("game_over", False) and not game["player_attacks"] and not game["ai_attacks"]
            if cols[c].button(label, key=f"p_{r}_{c}", disabled=not can_click, use_container_width=True):
                if on_click_cell:
                    on_click_cell(r, c)

def render_ai_board(game, on_click_attack):
    st.subheader("AI Board (click to strike)")
    disabled_all = game["game_over"] or game["turn"] != "player" or game.get("reposition", False)
    for r in range(BOARD_SIZE):
        cols = st.columns(BOARD_SIZE, gap="small")
        for c in range(BOARD_SIZE):
            cell = game["ai_grid"][r][c]
            attacked = cell in ("H", "M")
            ship_name = game["ai_ship_map"][r][c]
            if ship_name and game["ai_health"][ship_name] == 0:
                label = "💥"  # reveal entire sunk ship
            elif cell == "H":
                label = "💣"
            elif cell == "M":
                label = "💧"
            else:
                label = " "
            disabled = disabled_all or attacked
            if cols[c].button(label, key=f"a_{r}_{c}", disabled=disabled, use_container_width=True):
                on_click_attack(r, c)

def handle_place_click(game, r: int, c: int):
    if not game.get("placing") or game["game_over"]:
        return
    sel = game.get("placement_selected")
    if not sel or sel not in SHIP_TYPES:
        return
    size = SHIP_TYPES[sel]
    horizontal = game.get("placement_orientation", "H") == "H"
    # Validate
    if not can_place_ship(game["player_ship_map"], sel, size, r, c, horizontal):
        game["log"].append(f"Cannot place {sel} at {coords_to_label(r,c)} ({'H' if horizontal else 'V'}).")
        return
    # Place
    place_ship_on_board(game["player_grid"], game["player_ship_map"], sel, size, r, c, horizontal)
    # Update remaining
    remaining = [s for s in game["placement_remaining"] if s != sel]
    game["placement_remaining"] = remaining
    game["placement_selected"] = remaining[0] if remaining else None
    # Finish placing if done
    if not remaining:
        game["placing"] = False
        game["log"].append("All ships placed. Ready to play!")

# --- Actions ---
def handle_player_attack(game, r: int, c: int):
    if game["game_over"] or game["turn"] != "player":
        return
    res, sunk_name = perform_attack(game["ai_grid"], game["ai_ship_map"], game["ai_health"], r, c)
    coord_label = coords_to_label(r, c)
    if res == "repeat":
        game["log"].append(f"You already targeted {coord_label}. Pick another cell.")
        return
    game["player_attacks"].add((r, c))
    # Consolidated, single-line move log
    if res == "miss":
        move_text = f"Your shot at {coord_label}: miss."
    elif res == "hit":
        move_text = f"Your shot at {coord_label}: hit."
    elif res == "sunk":
        move_text = f"Your shot at {coord_label}: sunk {sunk_name}."
    else:
        move_text = f"Your shot at {coord_label}."
    game["move_no"] += 1
    game["moves"].append({"n": game["move_no"], "text": move_text})

    if all_sunk(game["ai_health"]):
        game["game_over"] = True
        game["winner"] = "player"
        game["log"].append("You win! All AI ships have been sunk.")
    else:
        game["turn"] = "ai"
        game["ai_should_move"] = True
        # Immediately rerun so the AI board shows this shot before AI moves
        st.rerun()

def perform_ai_turn(game, api_key: Optional[str], model: str):
    if game["game_over"] or game["turn"] != "ai":
        return
    # Compile AI history and tried coords
    tried = [coords_to_label(r, c) for (r, c) in sorted(list(game["ai_attacks"]))]
    ai_move = ai_pick_coordinate(api_key, model, game["ai_history"], tried)
    if ai_move is None:
        game["log"].append("AI cannot find a valid move. You win by default.")
        game["game_over"] = True
        game["winner"] = "player"
        game["turn"] = "player"
        game["ai_should_move"] = False
        return

    r, c = ai_move
    label = coords_to_label(r, c)
    # Ensure no repeats (guard)
    if (r, c) in game["ai_attacks"]:
        # Fallback to random unique coordinate
        all_cells = [(rr, cc) for rr in range(BOARD_SIZE) for cc in range(BOARD_SIZE)]
        remaining = [t for t in all_cells if t not in game["ai_attacks"]]
        if not remaining:
            game["log"].append("AI cannot find a valid move. You win by default.")
            game["game_over"] = True
            game["winner"] = "player"
            game["turn"] = "player"
            game["ai_should_move"] = False
            return
        r, c = random.choice(remaining)
        label = coords_to_label(r, c)

    res, sunk_name = perform_attack(game["player_grid"], game["player_ship_map"], game["player_health"], r, c)
    game["ai_attacks"].add((r, c))

    result_str = "miss"
    if res == "miss":
        move_text = f"AI shot at {label}: miss."
        result_str = "miss"
    elif res == "hit":
        move_text = f"AI shot at {label}: hit."
        result_str = "hit"
    elif res == "sunk":
        move_text = f"AI shot at {label}: sunk {sunk_name}."
        result_str = "sunk"
    else:
        move_text = f"AI shot at {label}."
    game["move_no"] += 1
    game["moves"].append({"n": game["move_no"], "text": move_text})

    game["ai_history"].append({"coord": label, "result": result_str})

    if all_sunk(game["player_health"]):
        game["game_over"] = True
        game["winner"] = "ai"
        game["log"].append("AI wins! All your ships have been sunk.")
    else:
        game["turn"] = "player"
    game["ai_should_move"] = False
    # Force a rerun so the player's board immediately reflects AI's shot
    st.rerun()

# --- Streamlit App ---
st.set_page_config(page_title="Cognition Battleship", layout="wide")

# Center and sharpen emojis/text inside buttons (applies to grid tiles)
st.markdown(
    """
    <style>
    .stButton > button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.35rem 0.35rem;
        line-height: 1.1;
        font-size: 1.35rem; /* larger for readability */
        min-height: 2.4rem; /* consistent tile height */
    }

    /* Outline emojis/text differently for light vs dark modes */
    @media (prefers-color-scheme: light) {
      .stButton > button {
        text-shadow:
          0 0 1px rgba(0,0,0,0.6),
          0 1px 0 rgba(0,0,0,0.3),
          -1px 0 0 rgba(0,0,0,0.3),
          1px 0 0 rgba(0,0,0,0.3);
      }
    }
    @media (prefers-color-scheme: dark) {
      .stButton > button {
        text-shadow: none; /* revert to previous look in dark mode */
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "game" not in st.session_state:
    st.session_state.game = new_game_state()

game = st.session_state.game

st.title("Cognition Battleship")
st.caption("Play Battleship against an AI. Click on the AI board to fire.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    # OpenAI configuration is implicit: API key from .env file
    load_dotenv()
    effective_api_key = os.getenv("OPENAI_API_KEY")

    if st.button("New Game / Reset", type="primary"):
        st.session_state.game = new_game_state()
        game = st.session_state.game
        st.rerun()

    # Allow randomizing the player's ships at the start of the game
    at_beginning = (not game["player_attacks"] and not game["ai_attacks"] and not game["game_over"])
    if st.button("Randomize My Ships", help="Re-roll your ship placement before the first move.", disabled=not at_beginning):
        player_grid, player_ship_map, player_health = place_ships_randomly()
        game["player_grid"] = player_grid
        game["player_ship_map"] = player_ship_map
        game["player_health"] = player_health
        st.rerun()

    # Reposition Ships (pre-game only)
    st.markdown("---")
    st.subheader("Reposition Ships")
    can_repos = at_beginning
    col1, col2 = st.columns(2)
    with col1:
        start_repos = st.button("Start", disabled=not can_repos or game.get("reposition", False))
    with col2:
        stop_repos = st.button("Finish", disabled=not game.get("reposition", False) or game.get("reposition_pick") is not None)

    if start_repos:
        game["reposition"] = True
        game["reposition_pick"] = None
        game["reposition_orientation"] = "H"
        st.rerun()
    if stop_repos:
        game["reposition"] = False
        game["reposition_pick"] = None
        st.rerun()

    if game.get("reposition", False):
        game["reposition_orientation"] = st.radio(
            "Orientation", options=["H", "V"], horizontal=True,
            index=0 if game.get("reposition_orientation", "H") == "H" else 1,
        )
        st.caption("Pick up a ship by clicking it on your board, then click a destination cell to drop it.")


# Layout: two columns for boards with a vertical divider
left, mid, right = st.columns([1, 0.03, 1])
def handle_reposition_click(game, r: int, c: int):
    if not game.get("reposition", False) or game.get("game_over", False):
        return
    # Only pre-move
    if game["player_attacks"] or game["ai_attacks"]:
        return
    pick = game.get("reposition_pick")
    if not pick:
        name = game["player_ship_map"][r][c]
        if not name:
            return
        ship_name, cells, orient = get_ship_cells(game["player_ship_map"], r, c)
        if not cells:
            return
        # Remove from board
        for (rr, cc) in cells:
            game["player_ship_map"][rr][cc] = None
            game["player_grid"][rr][cc] = ""
        game["reposition_pick"] = {"name": ship_name, "size": len(cells)}
        # Default orientation to current orientation of that ship
        game["reposition_orientation"] = orient
        st.rerun()
    else:
        name = pick["name"]
        size = pick["size"]
        horizontal = game.get("reposition_orientation", "H") == "H"
        if can_place_ship(game["player_ship_map"], name, size, r, c, horizontal):
            place_ship_on_board(game["player_grid"], game["player_ship_map"], name, size, r, c, horizontal)
            game["reposition_pick"] = None
            st.rerun()
        else:
            # Invalid drop; keep holding
            pass

with left:
    render_player_board(game, lambda rr, cc: handle_reposition_click(game, rr, cc) if game.get("reposition") else None)
with mid:
    st.markdown(
        "<div style='width:1px;height:100%;background-color:#e0e0e0;margin:0 auto;'></div>",
        unsafe_allow_html=True,
    )
with right:
    render_ai_board(game, lambda r, c: handle_player_attack(game, r, c))

# Process AI move if requested
if game["ai_should_move"] and not game["game_over"] and game["turn"] == "ai":
    perform_ai_turn(game, effective_api_key, DEFAULT_MODEL)

# Status area
st.markdown("---")
st.subheader("Game Status")
turn_text = "Game over" if game["game_over"] else f"Turn: {game['turn'].capitalize()}"
winner_text = f" • Winner: {game['winner'].capitalize()}" if game["game_over"] and game["winner"] else ""
st.write(f"{turn_text}{winner_text}")

st.subheader("Log")
if not game["moves"]:
    st.info("No events yet. Click a cell on the AI board to fire.")
else:
    total = len(game["moves"])  # 1..total
    start = max(0, total - 20)
    # Iterate from newest (index total-1) to oldest within the last 20
    for i in range(total - 1, start - 1, -1):
        e = game["moves"][i]
        st.write(f"{i + 1}. {e['text']}")

