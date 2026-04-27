# CliniMatch — How to Run (Step by Step)

## What You Need

| Service | Cost | API Key Needed? |
|---------|------|-----------------|
| OpenAI (GPT-4.1 mini) | ~$5-10 | YES — only paid API |
| ClinicalTrials.gov | FREE | NO |
| Semantic Scholar | FREE | NO |
| PubMed | FREE | NO |
| Streamlit | FREE | NO |

**Total cost: ~$5-10 for OpenAI credits (that's it)**

---

## STEP 1: Get OpenAI API Key (5 minutes)

1. Open https://platform.openai.com/signup
2. Create account (or log in)
3. Go to https://platform.openai.com/api-keys
4. Click **"Create new secret key"**
5. Name it anything (e.g., "clinimatch")
6. **COPY the key** — it starts with `sk-...` — save it somewhere, you can't see it again
7. Go to https://platform.openai.com/settings/organization/billing
8. Click **"Add payment method"** → add card → add **$10 credit**

---

## STEP 2: Open Terminal

**Option A — Command Prompt:**
- Press `Windows key`, type `cmd`, press Enter

**Option B — Git Bash (recommended):**
- Right-click on Desktop → "Open Git Bash Here"

---

## STEP 3: Set Your API Key

**In Command Prompt:**
```cmd
set OPENAI_API_KEY=sk-paste-your-key-here
```

**In Git Bash:**
```bash
export OPENAI_API_KEY="sk-paste-your-key-here"
```

> IMPORTANT: Replace `sk-paste-your-key-here` with your actual key. Keep quotes in Git Bash.

---

## STEP 4: Navigate to Project Folder

```cmd
cd "C:\Users\shaik\Desktop\ai fro engineers\project\clinical-trial-matcher"
```

---

## STEP 5: Run the App

**Try this first:**
```cmd
streamlit run app.py
```

**If "streamlit not found", use this instead:**
```cmd
"C:\Users\shaik\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py
```

---

## STEP 6: Use the App in Browser

The terminal will show:
```
Local URL: http://localhost:8501
```

Your browser opens automatically. If not, open http://localhost:8501

### What to do:
1. **Left sidebar** → "Load Sample Patient" is already selected
2. Pick a patient from the dropdown:
   - **Sarah Martinez** — Breast Cancer (Stage IIIA)
   - **James Chen** — Renal Cell Carcinoma (Stage IV)
   - **Maria Johnson** — Lung Cancer NSCLC (Stage IIIB)
3. Scroll down in sidebar → see **Weight Sliders** (leave defaults or adjust)
4. Click the big blue button: **"🔍 Find Matching Trials"**
5. Wait 30-60 seconds — you'll see progress:
   - ✅ Searching ClinicalTrials.gov...
   - ✅ Evaluating eligibility with AI...
   - ✅ Checking drug evidence...
   - ✅ Ranking complete!
6. **Results appear** — trial cards sorted by match score
7. Click **any trial card** → expand to see:
   - **Scores tab** — 5 objective scores
   - **Eligibility tab** — topic-wise breakdown (demographics, disease, labs, etc.)
   - **Evidence tab** — Semantic Scholar papers + PubMed papers + AI summary
   - **Locations tab** — trial sites with map

### To try different weights:
- Move the sliders (e.g., push "Geographic Proximity" to max)
- Click "Find Matching Trials" again
- Watch the ranking order change!

### To enter a custom patient:
- Switch to "Enter Manually" in the sidebar
- Fill in the fields
- Click "Find Matching Trials"

---

## STEP 7: Stop the App

Go back to terminal and press **Ctrl + C**

---

## All Commands (Copy-Paste Version)

### Command Prompt (full sequence):
```cmd
set OPENAI_API_KEY=sk-paste-your-key-here
cd "C:\Users\shaik\Desktop\ai fro engineers\project\clinical-trial-matcher"
"C:\Users\shaik\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py
```

### Git Bash (full sequence):
```bash
export OPENAI_API_KEY="sk-paste-your-key-here"
cd "/c/Users/shaik/Desktop/ai fro engineers/project/clinical-trial-matcher"
/c/Users/shaik/AppData/Local/Programs/Python/Python310/python.exe -m streamlit run app.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **"streamlit: not found"** | Use full path: `"C:\Users\shaik\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py` |
| **"OPENAI_API_KEY not set"** | You must set the key in the SAME terminal window. Run `set OPENAI_API_KEY=sk-...` again |
| **"AuthenticationError"** | Your key is wrong OR has no credit. Go to platform.openai.com → Billing → add $10 |
| **"No trials found"** | Condition too specific. Use a sample patient instead |
| **App is slow (30-60 sec)** | Normal — it calls OpenAI for each trial. 50 trials × 1 sec each |
| **Port 8501 already in use** | Run: `streamlit run app.py --server.port 8502` |
| **"ModuleNotFoundError"** | Run: `"C:\Users\shaik\AppData\Local\Programs\Python\Python310\python.exe" -m pip install streamlit openai requests biopython geopy pandas` |

---

## Cost Breakdown Per Search

| What Happens | API | Cost |
|-------------|-----|------|
| Extract disease from patient text | OpenAI | ~$0.001 |
| Search recruiting trials | ClinicalTrials.gov | FREE |
| Evaluate eligibility (per trial) | OpenAI | ~$0.01 |
| Search academic papers | Semantic Scholar | FREE |
| Search PubMed papers | PubMed | FREE |
| Generate evidence summary | OpenAI | ~$0.002 |
| **Total per search (50 trials)** | | **~$0.50** |

You can run ~20 full searches with $10 credit.
