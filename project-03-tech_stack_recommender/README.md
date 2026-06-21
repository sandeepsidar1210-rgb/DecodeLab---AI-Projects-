# DecodeLabs — Project 3: AI Recommendation Logic
## Tech Stack Recommender | Batch 2026

---

## 📁 Project Structure

```
project3/
│
├── recommender.py      ← Main script (run this)
├── raw_skills.csv      ← Dataset of job roles and skills
├── requirements.txt    ← Python libraries needed
└── README.md           ← This file
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the recommender
```bash
python recommender.py
```

---

## 🧠 How It Works

```
User Skills Input
      │
      ▼
TF-IDF Vectorization  ←── raw_skills.csv (job roles dataset)
      │
      ▼
Cosine Similarity Scoring (user vector vs every job role vector)
      │
      ▼
Sort Descending by Score
      │
      ▼
Return Top 3 Matches with:
  ✅ Match score (0% to 100%)
  ✅ Matched skills
  📌 Skill gaps (what to learn next)
```

---

## 🧪 Sample Test Inputs

| Skills You Enter | Expected Top Result |
|---|---|
| Python, SQL, Machine Learning, Statistics | Data Scientist |
| AWS, Docker, Kubernetes, Linux | DevOps Engineer |
| JavaScript, React, HTML, CSS | Frontend Developer |
| TensorFlow, PyTorch, Deep Learning, Python | ML Engineer |
| SQL, Power BI, Tableau, Excel | Business Intelligence |
| Solidity, Ethereum, Web3, Cryptography | Blockchain Developer |

---

## 📌 Key Concepts Used

- **Content-Based Filtering** — matches user profile to item attributes
- **TF-IDF Vectorization** — weights specific skills higher than generic ones
- **Cosine Similarity** — measures directional alignment between vectors (magnitude-invariant)
- **Top-N Filtering** — prevents choice overload, returns only best matches
- **IPO Model** — Input (user skills) → Process (similarity math) → Output (ranked roles)

---

## 🔧 Customization

To add more job roles, simply add rows to `raw_skills.csv`:
```
New Role Name,Skill1 Skill2 Skill3 Skill4 Skill5
```

Note: Skills in the CSV are space-separated (not comma-separated).

---

Built for **DecodeLabs Industrial Training Kit | Batch 2026**
