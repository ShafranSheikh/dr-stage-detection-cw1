# Setup Guide — Kaggle, Mac environment and GitHub

**Project:** Diabetic Retinopathy Stage Detection (Computer Vision CW1)
**Written:** 18 September 2026 · Follow the parts in order: **A → B → C**.
Total time: about 1 hour, most of it waiting for downloads.

## What we are setting up and why

| Place | Purpose |
|---|---|
| **Kaggle** | Where the model is trained. Free GPU, and the 3,662 retina images are already on Kaggle's servers, so nothing large is downloaded to your Mac. |
| **Your Mac (M2)** | Where the demo app runs for the video, and where the code lives. |
| **GitHub** | A private backup and history of your code. It also gives evidence that the work is yours, built step by step. |

Your project folder already exists at:

```
/Users/mohamedshafran/Downloads/Computer vision/CW1_DR_Stage_Detection
```

It contains `README.md`, `.gitignore`, `requirements.txt` and empty folders for the notebooks, app, models, results, figures, report and docs.

---

# Part A — Kaggle (about 20 minutes)

### A1. Sign in
Go to [kaggle.com](https://www.kaggle.com) and sign in (or create a free account with your email).

### A2. Verify your phone number
1. Open [kaggle.com/settings](https://www.kaggle.com/settings).
2. Find **Phone Verification** and enter your mobile number.
3. Type in the code Kaggle sends by SMS.

**Why:** Kaggle only unlocks GPUs after phone verification. If the Accelerator menu is greyed out later, this is the reason.

### A3. Get access to the APTOS 2019 data
1. Open the competition: <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
2. Open the **Data** tab.
3. If a button appears asking you to join the competition or accept the rules (wording such as *"I Understand and Accept"* or *"Late Submission"*), click it.

**Why:** competition data can only be attached to a notebook after you accept that competition's rules. The competition is closed, so there is nothing to submit — we only use the images.

You should now be able to see the file list: `train.csv`, `test.csv`, `train_images/`, `test_images/`.

### A4. Create your first notebook
1. Click **Create** (top-left) → **New Notebook**.
2. Rename it (click the title, top-left) to: `dr-00-environment-check`.

### A5. Turn on the GPU and the internet
In the right-hand panel (open it with the **⚙ / Session options** button if hidden):

- **Accelerator:** `GPU T4 x2` (or `GPU P100` if T4 is unavailable)
- **Internet:** `On`
- **Language:** Python

**Why internet:** Keras downloads the pretrained EfficientNet weights from the internet the first time we build the model.

### A6. Attach the images
1. In the right panel click **+ Add Input**.
2. Choose the **Competitions** tab, search for *APTOS 2019 Blindness Detection*, and click **+**.
3. The data now appears under `/kaggle/input/aptos2019-blindness-detection`.

### A7. Run this check cell
Paste this into the first cell and run it (Shift+Enter):

```python
# dr-00-environment-check — confirms the Kaggle session is ready to use
import sys, os, glob
import tensorflow as tf
import keras
import pandas as pd

print("Python    :", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("Keras     :", keras.__version__)
print("GPUs      :", tf.config.list_physical_devices("GPU"))

DATA = "/kaggle/input/aptos2019-blindness-detection"
print("\nFiles in the dataset folder:")
for name in sorted(os.listdir(DATA)):
    print("   ", name)

train = pd.read_csv(f"{DATA}/train.csv")
print("\ntrain.csv rows:", len(train))
print(train.head())

print("\nImages per stage (0 = No DR ... 4 = Proliferative):")
print(train["diagnosis"].value_counts().sort_index())

print("\nTraining image files found:", len(glob.glob(f"{DATA}/train_images/*.png")))
```

**What a good result looks like:** a TensorFlow version number, a non-empty GPU list, `train.csv rows: 3662`, and 3,662 image files.

**Send me the output.** The TensorFlow version decides which version we install on your Mac in Part B.

### A8. Saving and quota
- **Save Version → Quick Save** stores the notebook as it is.
- **Save Version → Save & Run All** runs the whole notebook in the background and stores its output files. We use this for long training runs.
- The free GPU allowance is roughly **30 hours per week** and a session runs up to about 12 hours. Keep the accelerator on **None** while you are only writing code, and stop the session when you finish (**Session options → Stop session**).

### A9. API token (optional, useful later)
1. [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token** → `kaggle.json` downloads.
2. Keep it somewhere private on your Mac. **Never** put it in the project folder or on GitHub.

We will use it to download the trained model file from the terminal. Note: the Kaggle key that appears inside your old tumour-segmentation notebook belongs to someone else — do not use it.

---

# Part B — Mac environment (about 30 minutes)

Open **Terminal** (press ⌘+Space, type "Terminal", press Enter). Run the commands one at a time.

### B1. Check the machine
```bash
uname -m
sw_vers
```
Expect `arm64` and a ProductVersion of 12.0 or newer.

> **Paste commands without the explanations after them.** In zsh a `#` typed on the command line is not a comment — it is passed to the command as an argument, and the command fails.

### B2. Install Apple's command line tools (this also installs Git)
```bash
git --version
```
If macOS shows a popup offering to install the command line tools, click **Install** and wait. If the popup does not appear but Git is missing, run `xcode-select --install`.

### B3. Use the Python you already have (Anaconda)

Your module environment `Practical/nibm_env` was built from Anaconda's Python 3.13, so it should already be on this Mac:

```bash
which python3
/opt/anaconda3/bin/python3 --version
```

Expect a path inside `/opt/anaconda3` and `Python 3.13.x`. If that works, skip to B4 — no new Python is needed.

*Only if it does not exist* (Anaconda was removed), install Homebrew and Python with it:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install python@3.11
```
Then use `/opt/homebrew/bin/python3.11` wherever the steps below say `/opt/anaconda3/bin/python3`.

### B4. Do not reuse `Practical/nibm_env`

That environment was created inside `~/Personal/Lecture notes/…` and later copied into Downloads. A virtual environment stores the absolute path it was built at, so the copy's `activate` script and installed commands point at the old location. We build a fresh one for this assignment, using the same recipe as your module's `setup_env.sh`.

### B5. Move into the project folder
```bash
cd "/Users/mohamedshafran/Downloads/Computer vision/CW1_DR_Stage_Detection"
ls
```
The quotes matter because the path contains a space. You should see `README.md`, `requirements.txt` and the folders.

### B6. Create the virtual environment
```bash
/opt/anaconda3/bin/python3 -m venv venv
source venv/bin/activate
```
Your prompt now starts with `(venv)`. That means Python packages install into this project only, not into the whole system. This is the same pattern as your module's `setup_env.sh`.

To leave it later: `deactivate`. To come back: `cd` into the folder and run `source venv/bin/activate` again.

### B7. Install the packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This downloads roughly 1 GB, so it can take 5–15 minutes.

Then register this environment as a Jupyter kernel, exactly as your module's setup script does, so you can also open notebooks locally if you want to:
```bash
python -m ipykernel install --user --name=dr-cw1 --display-name="DR CW1"
```

**About versions:** your module environment already runs TensorFlow 2.21 with Keras 3.14, which is newer than the version Kaggle is likely to have. That direction is the safe one — a newer Keras can normally open a model saved by an older Keras 3. We confirm it with a real model early in the training phase. If it ever complains, we pin the local version to Kaggle's, for example `pip install "tensorflow==2.20.*"`.

### B8. Check that it works

This runs the OpenCV calls our preprocessing depends on, then saves and reloads a model exactly as the app will:

```bash
python - << 'EOF'
import numpy as np, cv2, tensorflow as tf, keras, pandas as pd, sklearn, streamlit
print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
print("OpenCV    ", cv2.__version__, "| NumPy", np.__version__, "| pandas", pd.__version__)

img   = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
back  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
blur  = cv2.GaussianBlur(back, (0, 0), 3)
sharp = cv2.addWeighted(back, 1.5, blur, -0.5, 0)
gray  = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
print("OpenCV pipeline OK:", sharp.shape, "| sharpness", round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1))

m = keras.Sequential([keras.layers.Input((8,)), keras.layers.Dense(5, activation="softmax")])
m.save("models/_smoke_test.keras")
print("save/load OK:", keras.models.load_model("models/_smoke_test.keras").output_shape)
EOF
rm models/_smoke_test.keras
```

Finally check Streamlit opens in your browser:
```bash
streamlit hello
```
Press **Ctrl+C** in Terminal to stop it.

**Note:** TensorFlow has no official GPU support on macOS, so it uses the processor. That is fine — the app only predicts one image at a time.

---

# Part C — GitHub (about 20 minutes)

If you already have a GitHub account with SSH set up, jump to **C4**.

### C1. Account
Create or sign in at [github.com](https://github.com).

### C2. Tell Git who you are
```bash
git config --global user.name "Mohamed Shafran"
git config --global user.email "shafransheikh@gmail.com"
git config --global init.defaultBranch main
```

### C3. Create an SSH key so Git can push without passwords
```bash
ssh-keygen -t ed25519 -C "shafransheikh@gmail.com"
```
Press Enter to accept the default file name. A passphrase is optional (press Enter twice to skip).

```bash
eval "$(ssh-agent -s)"
touch ~/.ssh/config
open -e ~/.ssh/config
```
In the window that opens, add these four lines, then save and close:
```
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub
```
The second line copies the public key to your clipboard.

On GitHub: **profile photo → Settings → SSH and GPG keys → New SSH key**. Title it `MacBook M2`, paste into the Key box, click **Add SSH key**.

Test it:
```bash
ssh -T git@github.com
```
Type `yes` if it asks about authenticity. Success looks like: *"Hi yourusername! You've successfully authenticated, but GitHub does not provide shell access."*

### C4. Create the repository
On GitHub: **+ (top right) → New repository**.

- **Repository name:** `dr-stage-detection-cw1`
- **Visibility:** **Private** ← important
- Do **not** tick "Add a README", ".gitignore" or a licence — the folder already has them.
- Click **Create repository**.

**Why private:** the coursework must be original work. A public repository can be copied by another student, and a similarity check would then flag your own work.

### C5. Push the project folder
```bash
cd "/Users/mohamedshafran/Downloads/Computer vision/CW1_DR_Stage_Detection"
git init
git add .
git commit -m "Set up project structure and planning documents"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/dr-stage-detection-cw1.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your GitHub username. Refresh the repository page — your files should be there.

### C6. The routine from now on
Every time a piece of work is finished (a notebook runs, the app changes, a figure is produced):

```bash
cd "/Users/mohamedshafran/Downloads/Computer vision/CW1_DR_Stage_Detection"
git add -A
git commit -m "Add data exploration notebook and class distribution figure"
git push
```

To get a notebook out of Kaggle and into the repository: in the Kaggle notebook use **File → Download notebook**, then move the `.ipynb` file into `kaggle_notebooks/` and commit it.

Write short, plain commit messages that say what changed, for example:
- `Add retina cropping and CLAHE preprocessing`
- `Train baseline EfficientNetB0 and log validation results`
- `Fix wrong input range for EfficientNet`

**Never commit:** `kaggle.json`, the dataset images, or trained model files. The `.gitignore` already blocks them. GitHub warns above 50 MB per file and refuses anything above 100 MB.

---

# Part D — Finished when all of these are true

- [ ] Kaggle phone verified; Accelerator menu is not greyed out
- [ ] APTOS competition rules accepted; data visible under `/kaggle/input/aptos2019-blindness-detection`
- [ ] The check cell printed `train.csv rows: 3662` and a GPU in the list
- [ ] Terminal shows `(venv)` and `python -c "import tensorflow ..."` prints a version
- [ ] The save/load test printed `save/load OK`
- [ ] `ssh -T git@github.com` greets you by username
- [ ] The private repository shows your folders after the first push

---

# Part E — Errors you may hit, and what they mean

| Message | Cause | Fix |
|---|---|---|
| `usage: uname [-amnoprsv]` (or similar) right after pasting | zsh passed the `#` note as an argument — it is not a comment on the command line | Paste the command on its own |
| `FileNotFoundError: /kaggle/input/<name>` | The data sits one level deeper (`/kaggle/input/competitions/<slug>`) | Find the folder that contains `train.csv` instead of hard-coding a path |
| Kaggle Accelerator options greyed out | Phone not verified | Part A2 |
| `zsh: command not found: brew` | Homebrew is installed but not on the PATH | Re-run the two `shellenv` lines in B3 |
| `error: externally-managed-environment` during pip install | You are installing outside the virtual environment | Run `source venv/bin/activate` first (prompt must show `(venv)`) |
| `zsh: no such file or directory: /Users/.../Computer` | The path has a space and was not quoted | Wrap the whole path in double quotes |
| `Permission denied (publickey)` on push | The SSH key is not on GitHub, or not loaded | Redo C3, then `ssh -T git@github.com` |
| `remote: error: File ... exceeds GitHub's file size limit` | A model or data file got committed | Remove it from the commit; check `.gitignore` |
| `error: failed to push some refs` | GitHub has commits your Mac does not | `git pull --rebase origin main`, then push again |
| TensorFlow prints messages about oneDNN or AVX on start-up | Normal information messages | Ignore them |
| Kaggle: "Your notebook tried to allocate more memory than is available" | Batch size or image size too large | Lower the batch size (we will tune this during training) |

---

# Part F — What happens next

Once Part A7's output is in, I will:

1. Write **notebook 1 — data exploration and splitting** as a complete, commented file, saved into `kaggle_notebooks/`.
2. Explain the two new ideas it uses (finding duplicate images, and keeping duplicate groups together when splitting) before you run it.
3. Give you the steps to import and run it on Kaggle.
