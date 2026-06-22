# 🤖 DecodeLabs AI Project 1: LogicBot

**Author:** Sandeep Kumar Sidar  
**Institution:** International Institute of Information Technology (IIIT), Naya Raipur  
**Batch:** 2026 | DecodeLabs Artificial Intelligence Track

---

## 📌 Overview
This repository contains the first milestone of the DecodeLabs AI Industrial Training. **LogicBot** is a deterministic, rule-based conversational agent built on pure logic. It serves as an introduction to how machines process textual inputs and execute predefined actions before moving on to complex machine learning models.

## 🚀 Key Features
* **Input Sanitization:** Automatically cleans and normalizes user input (lowercasing and stripping whitespace) for highly reliable processing.
* **O(1) Intent Matching:** Utilizes Python dictionary lookups (`responses.get()`) to instantly route user queries to the correct responses.
* **Graceful Fallbacks:** Equipped with a fallback mechanism to handle unrecognized or out-of-scope queries seamlessly.
* **Continuous Execution:** Runs on an infinite `while` loop to maintain conversational context until the user explicitly triggers shutdown commands (`exit`, `quit`).

## 🛠️ Tech Stack
* **Language:** Pure Python
* **Architecture:** Rule-based logic (If-Else & Dictionary Routing)

## 🏃‍♂️ How to Run
Run the script in your terminal to start chatting:
```bash
python chatbot.py