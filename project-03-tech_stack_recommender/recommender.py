import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #convert text into numerical vector
from sklearn.metrics.pairwise import cosine_similarity 

#mathematical similarity between two vectors by measuring the angle between them
import os 
#It allows the script to interact with your computer's operating system. In your recommender.py file, it is specifically used to clear the terminal screen (os.system('cls' if os.name == 'nt' else 'clear')) so the user interface looks clean and readable when running the application.

import sys 
#It provides access to variables and functions that interact closely with the Python interpreter. In your project, it is used for error handling—specifically to gracefully stop the program using sys.exit(1) if the application fails to find the required raw_skills.csv dataset.


#  DISPLAY HELPERS

def print_banner():
    print("\n" + "-" * 60)
    print("   🤖  DecodeLabs | Project 3 — AI Recommendation Logic")
    print("        Tech Stack & Career Path Recommender")
    print("-" * 60)

def print_score_bar(score, width=20):
    filled = int(score * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    # It builds the actual visual bar.

    return f"[{bar}] {score:.1%}"
    # It outputs the final string, wrapping the text bar in brackets and converting the decimal score into a clean percentage format (like 80.0%).

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


#  STEP 1 — LOAD DATASET

def load_dataset(filepath="raw_skills.csv"):
    """Load the job roles and their required skills from CSV."""
    try:
        df = pd.read_csv(filepath)
        print(f"  ✅ Dataset loaded: {len(df)} job roles found.\n")
        return df
    except FileNotFoundError:
        print(f"\n  ❌ ERROR: '{filepath}' not found.")
        print("    Make sure raw_skills.csv is in the same folder as recommender.py")
        sys.exit(1)
        # The line sys.exit(1) is used to immediately stop the program from running when a critical error occurs.


# STEP 2 — BUILD TF-IDF MODEL

def build_tfidf_model(df):
    """ Fit TF-IDF vectorizer on job role skills.
    
    - TF  (Term Frequency)         : How often a skill appears in a role
    - IDF (Inverse Doc Frequency)  : Penalizes skills common to all roles
    - Result: Specific/rare skills get higher weights """

    vectorizer = TfidfVectorizer()
    # Creates an empty instance (a blank template) of the TF-IDF Vectorizer tool

    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    # Fit: It reads through all the skills in your dataset column (df["skills"]) to learn every unique skill name that exists (building its master vocabulary).

    # Transform: It converts those text descriptions into a mathematical table of numbers (a matrix) using the TF-IDF scoring rules.

    print(f"  ✅ TF-IDF model built.")
    print(f"     Vocabulary size : {len(vectorizer.vocabulary_)} unique skills")
    # Counts and displays the total number of unique technical skills found across your entire CSV dataset.

    print(f"     Matrix shape    : {tfidf_matrix.shape} (roles x skills)\n")
    # Prints the dimensions (rows and columns) of your newly created math table.

    return vectorizer, tfidf_matrix


#  STEP 3 — GET USER INPUT

def get_user_skills():
    """Collect and validate minimum 3 skills from the user."""
    print("─" * 60)
    print("  📝  ENTER YOUR SKILLS")
    print("─" * 60)
    print("  Type your skills separated by commas.")
    print("  Minimum 3 skills required for accurate matching.\n")
    print("  Examples:")
    print("    → Python, Machine Learning, SQL, Statistics")
    print("    → AWS, Docker, Kubernetes, Linux")
    print("    → JavaScript, React, HTML, CSS, NodeJS\n")

    while True:
        raw_input = input("  Your skills: ").strip()

        if not raw_input:
            print("  ⚠️  Input cannot be empty. Please try again.\n")
            continue

        # Clean and normalize
        skills = [s.strip().title() for s in raw_input.split(",")]
        # It splits the long user input string wherever there is a comma, cuts off accidental spaces around the words, and capitalizes the first letter of every skill.

        skills = [s for s in skills if s]  # remove empty entries

        # This prevents accidental extra commas from breaking your code. If a user types "Python, SQL, ", the split action would create an accidental empty item at the end: ['Python', 'SQL', '']. This line checks if s (which means "if the skill actually contains text"). If an item is blank, it gets tossed out, leaving you with a perfectly clean list: ['Python', 'SQL'].

        if len(skills) < 3:
            print(f"  ⚠️  You entered {len(skills)} skill(s). Minimum 3 required.\n")
            continue

        print(f"\n  ✅ Skills received: {', '.join(skills)}\n")
        # It takes your individual cleaned list items (like ['Python', 'SQL', 'AWS']) and stitches them back together into a single readable sentence separated by commas (Python, SQL, AWS). This gives the user instant visual feedback that the application correctly understood exactly what they typed.

        return skills


# STEP 4 — VECTORIZE USER PROFILE

def vectorize_user(vectorizer, user_skills):
    """ Convert user skills into a TF-IDF vector.
    MUST use the same vectorizer (same vocabulary space)
    as the job roles — otherwise similarity math breaks."""

    user_string = " ".join(user_skills)
    # The text processing tool (TfidfVectorizer) is built to read sentences or paragraphs of text, not Python lists.

    user_vector = vectorizer.transform([user_string])
    # It passes that text sentence into the exact same TF-IDF model built from your job dataset, converting the text into a mathematical vector (a matrix of numbers).

    return user_vector


# STEP 5 — COMPUTE COSINE SIMILARITY

def compute_similarity(user_vector, tfidf_matrix):
    """
    Cosine Similarity = (A · B) / (||A|| x ||B||)
    
    Measures the ANGLE between two vectors — not magnitude.
    This makes it immune to document length differences.
    Score range: 0 (no match) to 1 (perfect match)
    """

    scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    # It compares the user's skill vector against every single job role vector in your dataset all at once, calculates their similarity, and grabs the resulting row of scores.


    # The cosine_similarity function measures the angle between vectors, giving a score between 0.0 (no matching skills) and 1.0 (a perfect skill match). Because cosine_similarity naturally outputs a nested 2D list (like [[0.85, 0.42, 0.0, ...]]), adding [0] at the end flattens it into a simple 1D list of scores so it's easier to work with.

    return scores


#  STEP 6 — SORT AND FILTER (Top-N Pipeline)

def get_top_recommendations(df, scores, top_n=3):
    """
    Pipeline Steps 3 & 4:
    - Sort all roles by similarity score (descending)
    - Filter: return only Top-N to prevent choice overload
    """

    top_indices = scores.argsort()[::-1][:top_n]
    # scores.argsort(): Instead of sorting the actual match percentages, it tells you the original index positions of those scores from lowest to highest.

    # [::-1]: This is a Python slicing trick that reverses the list. Now, the index positions are ordered from the highest score to the lowest score.

    # [:top_n]: This slices the list to keep only the top requested number of results (which is set to top_n = 3).

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        # The enumerate(..., start=1) function acts like a counter, keeping track of the loop's current turn—assigning 1 to the first best match, 2 to the second, and 3 to the third.

        role = df.iloc[idx]["job_role"]
        # It looks at your main spreadsheet dataset (df) at the specific row number (idx) and grabs the text inside the "job_role" column.

        score = scores[idx]
        # It pulls out the exact mathematical match percentage calculated earlier for this specific job row.

        role_skills = df.iloc[idx]["skills"].lower().split()
        # It looks up the required skills for that job row, forces all the text into lowercase using .lower(), and chops the sentence into an array of individual words using .split().


        # It prepares the job's skills for a background comparison. By making everything lowercase and splitting it into a list, your script can easily cross-reference it against the user's input to calculate skill gaps and matching tags.

        results.append({
            "rank"        : rank,
            "role"        : role,
            "score"       : score,
            "role_skills" : role_skills,
            "index"       : idx
        })
        # It takes all the separate pieces of data we just gathered (rank, role name, score, list of skills, and original row index) and packs them tightly into a single dictionary "package". It then slides that package into your empty results container list.

    return results


# STEP 7 — DISPLAY RESULTS

def display_results(results, user_skills, df):
    """Show the top recommended roles with scores, matches, and skill gaps."""

    print("═" * 60)
    print("   🎯  TOP RECOMMENDED CAREER PATHS FOR YOU")
    print("═" * 60)

    if results[0]["score"] == 0.0:
        # It checks if the highest-ranked job recommendation (the #1 result at index 0) has a matching score of exactly 0.0
        print("\n  ⚠️  No matches found!")
        print("  Your skills didn't align with any known tech role.")
        print("  Try using standard tech terms like 'Python', 'AWS', 'React'.\n")
        return

    user_set = set([s.lower() for s in user_skills])
    # A set in Python is a special collection that ignores duplicates and allows for ultra-fast matching.

    for r in results:
        rank      = r["rank"]
        role      = r["role"]
        score     = r["score"]
        role_set  = set(r["role_skills"])

        matched_skills = user_set.intersection(role_set)
        # It uses a mathematical operation called intersection to find only the skills that exist in both the user_set and the role_set.

        # It extracts your overlapping strengths. If your skills are {'python', 'sql', 'aws'} and the job requires {'python', 'sql', 'java', 'c++'}, the intersection instantly figures out that your matching skills are {'python', 'sql'} so they can be highlighted in green on the dashboard.

        missing_skills = role_set - user_set
        # This performs your automated skill gap analysis.

        # Pick top 3 missing skills to show as growth areas
        top_missing = list(missing_skills)[:3]
        # converts it back into a standard Python list, and slices it to grab only the first 3 items.

        # [:3] keeps the interface clean by suggesting just the top 3 key growth areas for you to focus on next.

        medal = ["🥇", "🥈", "🥉"][rank - 1]
        # Python lists start counting at index 0. Because your ranking counter starts at 1 (rank = 1 for the best match), subtracting 1 (rank - 1) maps the ranks perfectly to the list:
        # Rank 1 becomes index 0, which pulls 🥇
        # Rank 2 becomes index 1, which pulls 🥈
        # Rank 3 becomes index 2, which pulls 🥉

        print(f"\n  {medal}  #{rank}  {role}")
        # It displays the final visual badge, the placement number, and the career path name all in one line.

        print(f"       Match Score  : {print_score_bar(score)}")
        # Instead of printing a raw, confusing math decimal like 0.8543, it displays the clean, visual shaded progress bar alongside the percentage (like [████████████████░░░░] 85.4%) directly underneath the job title.

        if matched_skills:
            print(f"       ✅ Your Match : {', '.join(sorted(matched_skills))}")
        else:
            print(f"       ✅ Your Match : Semantic / partial alignment")

        if top_missing:
            print(f"       📌 Skill Gaps : {', '.join(sorted(top_missing))} (learn these next!)")
            # This provides immediate, actionable career guidance. It tells the user exactly what specific technologies they should study next to become a competitive candidate for that particular role.

        print()
        # print() adds visual breathing room. Without it, the text blocks for Recommendation #1 and Recommendation #2 would merge together into a dense wall of text.

    print("─" * 60)


# ✅ FIX 2: show_all_scores was accidentally nested inside display_results.
#            It must be its own separate function at the top level (no indentation).

# BONUS — SHOW ALL SCORES (Optional)

def show_all_scores(df, scores):
    """Display similarity scores for all job roles, sorted."""

    print("\n  📊  FULL RANKING — All Job Roles")
    print("─" * 60)

    sorted_indices = scores.argsort()[::-1]
    # It sorts the index positions of all job roles in your dataset from the absolute highest score to the lowest score.

    for i, idx in enumerate(sorted_indices, start=1):
        # It loops through every single job role in the newly sorted order.

        role  = df.iloc[idx]["job_role"]
        score = scores[idx]
        bar   = print_score_bar(score, width=15)
        print(f"  {i:>2}. {role:<30} {bar}")
        # {i:>2} forces the list number to be right-aligned within 2 character spaces (keeping numbers 1 through 20 perfectly lined up).

        # {role:<30} forces the job title to be left-aligned and pads it with spaces out to exactly 30 characters wide.

        # It stops the output from looking jagged. Because job titles have different lengths (e.g., "DevOps Engineer" vs. "ML"), this layout acts like an invisible spreadsheet table so all your text bars stack in a perfect straight column on the right.

    print()


# MAIN — PIPELINE ORCHESTRATION

def main():
    clear_screen()
    print_banner()
    print()

    # ── Load Data ──
    print("  [1/4] Loading dataset...")
    df = load_dataset("raw_skills.csv")

    # ── Build TF-IDF Model ──
    print("  [2/4] Building TF-IDF model...")
    vectorizer, tfidf_matrix = build_tfidf_model(df)

    while True:
        # ── Get User Input ──
        print("  [3/4] User Input")
        user_skills = get_user_skills()

        # ── Vectorize + Score + Sort + Filter ──
        print("  [4/4] Running similarity engine...\n")
        user_vector = vectorize_user(vectorizer, user_skills)
        scores      = compute_similarity(user_vector, tfidf_matrix)
        results     = get_top_recommendations(df, scores, top_n=3)

        # ── Display Results ──
        display_results(results, user_skills, df)

        # ── Optional: Show full ranking ──
        show_all = input("  Show full ranking for all roles? (y/n): ").strip().lower()
        if show_all == "y":
            show_all_scores(df, scores)

        # ── Loop or Exit ──
        again = input("  Try again with different skills? (y/n): ").strip().lower()
        if again != "y":
            print("\n  👋  Thanks for using the Tech Stack Recommender!")
            print("      Built for DecodeLabs — Batch 2026\n")
            break

        print("\n" + "─" * 60 + "\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()