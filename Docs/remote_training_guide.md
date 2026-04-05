# Running Long Training Sessions on a Remote Server

To ensure your training doesn't stop when you disconnect or lose your internet connection, use one of the following methods.

## Option 1: Using `tmux` (Recommended)
`tmux` creates a persistent terminal session that stays alive on the server even when you log out.

1. **Start a new session:**
   ```bash
   tmux new -s training
   ```
2. **Start your Jupyter server or training script inside this session:**
   ```bash
   # If running a script
   python training_script.py
   
   # Or if starting Jupyter
   jupyter notebook --no-browser --port=8888
   ```
3. **Detach from the session:**
   Press `Ctrl + B`, then release and press `D`. You can now safely close your terminal/SSH connection.
4. **Reconnect and reattach later:**
   When you log back into the server, run:
   ```bash
   tmux attach -t training
   ```

## Option 2: Using `nohup`
`nohup` (No Hang Up) allows a command to ignore the signal sent when a terminal closes.

1. **Run your script in the background:**
   ```bash
   nohup python training_script.py > output.log 2>&1 &
   ```
   - This sends all output to `output.log`.
   - The `&` at the end puts it in the background.
2. **To check if it's still running:**
   ```bash
   ps aux | grep training_script.py
   ```

## Option 3: Converting Notebook to Script
If your training is in a notebook (`.ipynb`), it's often more robust to convert it to a `.py` script for long-running jobs.

1. **Convert the notebook:**
   ```bash
   jupyter nbconvert --to script training_setup.ipynb
   ```
2. **Run the resulting script using `tmux` or `nohup`** as shown above.

## Tip for Jupyter Notebook Users
If you are already running a cell in a notebook and afraid to close the browser:
- The cell execution happens on the *server*. As long as the Jupyter server process is alive, the cell will continue to run even if you close the browser tab.
- **Crucially:** Ensure the Jupyter server itself was started in a way that it won't die when you disconnect (like using `tmux` as described in Option 1).
