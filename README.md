📈 Job & Skill Demand Trend Prediction (ML Project)
🔍 Project Overview

This project analyzes real-world IT job postings to identify trending jobs and in-demand skills using Machine Learning.
It scrapes job data, cleans and preprocesses it, trains ML models with imbalance handling, and deploys predictions using Streamlit.

🎯 Objective

Identify high-demand IT jobs

Detect trending technical skills

Predict whether a job role is Trending or Not

Build an end-to-end ML pipeline from data collection to deployment

🧠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

SMOTE (imbalanced-learn)

TF-IDF (NLP)

Random Forest, Logistic Regression

Selenium (Web Scraping)

Streamlit (Deployment)

📂 Dataset

Source: Naukri.com

Domain: IT / Tech Jobs

Size: ~250+ job postings

Key Columns:

job_title

company

location

experience

inferred_skills

source

⚙️ Project Pipeline
1️⃣ Data Collection

Scraped IT job postings from Naukri

Extracted job title, company, location, experience

2️⃣ Data Cleaning & Preprocessing

Removed duplicates & null values

Parsed experience into numeric form

Inferred skills from job titles

Created skill_count feature

3️⃣ Target Engineering

Defined Job Demand based on skill richness

Jobs in top 25% skill count → Trending (1)

4️⃣ Feature Engineering

TF-IDF on:

Job Title

Skill List

Numeric features:

Minimum experience

Skill count

5️⃣ Model Training

Models used:

Logistic Regression

Random Forest (final)

Applied SMOTE to handle class imbalance

Achieved ~75–82% accuracy

6️⃣ Deployment

Built a Streamlit web app

Users input:

Job Title

Skills

Experience

Model predicts:

🔥 Trending / 📉 Not Trending

Confidence score

🖥️ Streamlit App Features

Real-time prediction

Clean UI

Uses saved ML pipeline (model.pkl)

No feature mismatch issues
