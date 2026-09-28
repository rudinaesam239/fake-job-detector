"""
Fake Job Posting Detector — Streamlit demo
-------------------------------------------
Paste (or fill in) a job ad and get a fraud-risk score from the model
trained in the companion notebook (Logistic Regression on TF-IDF text
+ metadata features, EMSCAD job-postings dataset).

Run with:
    streamlit run app.py

Needs, in the same folder:
    job_features.py        (feature engineering, shared with the notebook)
    fake_job_model.joblib  (the trained pipeline + tuned threshold)
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from job_features import build_features

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")

EMPLOYMENT_TYPES = ["Not specified", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
EXPERIENCE_LEVELS = ["Not specified", "Internship", "Entry level", "Associate", "Mid-Senior level",
                     "Director", "Executive", "Not Applicable"]
EDUCATION_LEVELS = ["Not specified", "Some High School Coursework", "High School or equivalent",
                    "Vocational", "Vocational - Degree", "Some College Coursework Completed",
                    "Associate Degree", "Bachelor's Degree", "Master's Degree", "Doctorate",
                    "Professional", "Certification", "Unspecified"]

EXAMPLES = {
    "-- Choose an example --": None,
    "Likely fake (work-from-home, upfront details)": dict(
        title="Data Entry Clerk - Work From Home",
        country="US", employment_type="Part-time", required_experience="Not specified",
        required_education="Not specified", industry="", function="",
        company_profile="", requirements="",
        benefits="",
        description=("Earn $5000 weekly working from home! No experience needed. "
                     "Send your bank details and a small registration fee by email to start today!!! "
                     "Limited positions available, apply now."),
        telecommuting=True, has_company_logo=False, has_questions=False, has_salary=False,
    ),
    "Likely real (detailed, ordinary ad)": dict(
        title="Junior Backend Developer",
        country="EG", employment_type="Full-time", required_experience="Entry level",
        required_education="Bachelor's Degree", industry="Computer Software", function="Engineering",
        company_profile=("We are a growing fintech company based in Cairo, building payment "
                         "infrastructure for small businesses across the region."),
        requirements=("1-2 years of experience with Python or Java. Familiarity with SQL and REST APIs. "
                      "Good communication skills and a degree in Computer Science or related field."),
        benefits="Health insurance, hybrid work, annual training budget.",
        description=("We are looking for a Junior Backend Developer to join our engineering team. "
                     "You will work on our core payments API, write tests, and collaborate with senior "
                     "engineers on code reviews. This is a great opportunity to grow in a fast-paced team."),
        telecommuting=False, has_company_logo=True, has_questions=True, has_salary=False,
    ),
}


@st.cache_resource
def load_model():
    return joblib.load("fake_job_model.joblib")


def score_posting(fields, bundle):
    row = pd.DataFrame([{
        "title": fields["title"],
        "location": fields["country"],
        "salary_range": "yes" if fields["has_salary"] else None,
        "company_profile": fields["company_profile"] or None,
        "description": fields["description"] or None,
        "requirements": fields["requirements"] or None,
        "benefits": fields["benefits"] or None,
        "telecommuting": int(fields["telecommuting"]),
        "has_company_logo": int(fields["has_company_logo"]),
        "has_questions": int(fields["has_questions"]),
        "employment_type": None if fields["employment_type"] == "Not specified" else fields["employment_type"],
        "required_experience": None if fields["required_experience"] == "Not specified" else fields["required_experience"],
        "required_education": None if fields["required_education"] == "Not specified" else fields["required_education"],
        "industry": fields["industry"] or None,
        "function": fields["function"] or None,
    }])
    feats = build_features(row)
    model = bundle["model"]
    score = model.decision_function(feats)[0] if bundle["score_type"] == "decision_function" \
        else model.predict_proba(feats)[0, 1]
    return float(score), row


def score_to_probability_like(score, threshold):
    """Map an unbounded decision-function score to a 0-100 display gauge, centred on the tuned threshold."""
    return float(100 / (1 + np.exp(-(score - threshold))))


# ---------------------------------------------------------------- UI
st.title("🕵️ Fake Job Posting Detector")
st.caption(
    "A traditional machine-learning demo (TF-IDF + Logistic Regression, no deep learning) trained on the "
    "EMSCAD job-postings dataset. Paste a real ad or try an example below."
)

bundle = load_model()
threshold = bundle["threshold"]

example_choice = st.selectbox("Load an example", list(EXAMPLES.keys()))
example = EXAMPLES[example_choice] or {}

with st.form("job_form"):
    st.subheader("Job details")
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Job title*", value=example.get("title", ""), placeholder="e.g. Marketing Intern")
        country = st.text_input("Country (2-letter code, optional)", value=example.get("country", ""), placeholder="e.g. US, EG, NG")
        employment_type = st.selectbox("Employment type", EMPLOYMENT_TYPES,
                                       index=EMPLOYMENT_TYPES.index(example.get("employment_type", "Not specified"))
                                       if example.get("employment_type") in EMPLOYMENT_TYPES else 0)
        required_experience = st.selectbox("Required experience", EXPERIENCE_LEVELS,
                                           index=EXPERIENCE_LEVELS.index(example.get("required_experience", "Not specified"))
                                           if example.get("required_experience") in EXPERIENCE_LEVELS else 0)
        required_education = st.selectbox("Required education", EDUCATION_LEVELS,
                                          index=EDUCATION_LEVELS.index(example.get("required_education", "Not specified"))
                                          if example.get("required_education") in EDUCATION_LEVELS else 0)
    with c2:
        industry = st.text_input("Industry (optional, free text)", value=example.get("industry", ""), placeholder="e.g. Computer Software")
        function = st.text_input("Job function (optional, free text)", value=example.get("function", ""), placeholder="e.g. Engineering")
        telecommuting = st.checkbox("Remote / work-from-home", value=example.get("telecommuting", False))
        has_company_logo = st.checkbox("Ad shows a company logo", value=example.get("has_company_logo", True))
        has_questions = st.checkbox("Ad includes screening questions", value=example.get("has_questions", True))
        has_salary = st.checkbox("Ad states a salary range", value=example.get("has_salary", False))

    company_profile = st.text_area("Company profile", value=example.get("company_profile", ""), height=80,
                                   placeholder="A short paragraph describing the company (optional but informative).")
    description = st.text_area("Job description*", value=example.get("description", ""), height=140,
                               placeholder="Paste the main text of the job ad here.")
    requirements = st.text_area("Requirements", value=example.get("requirements", ""), height=100)
    benefits = st.text_area("Benefits", value=example.get("benefits", ""), height=80)

    submitted = st.form_submit_button("Check this posting", use_container_width=True)

if submitted:
    if not title.strip() and not description.strip():
        st.warning("Please enter at least a job title or a description.")
    else:
        fields = dict(title=title, country=country, employment_type=employment_type,
                     required_experience=required_experience, required_education=required_education,
                     industry=industry, function=function, company_profile=company_profile,
                     description=description, requirements=requirements, benefits=benefits,
                     telecommuting=telecommuting, has_company_logo=has_company_logo,
                     has_questions=has_questions, has_salary=has_salary)
        score, row = score_posting(fields, bundle)
        is_fake = score >= threshold
        gauge = score_to_probability_like(score, threshold)

        st.divider()
        if is_fake:
            st.error(f"🔴 Likely **FAKE** — risk score {gauge:.0f} / 100")
        else:
            st.success(f"🟢 Likely **REAL** — risk score {gauge:.0f} / 100")
        st.progress(min(max(gauge / 100, 0.0), 1.0))
        st.caption(
            f"Raw model score: {score:.2f} (decision threshold: {threshold:.2f}). "
            "The risk score above is a rescaled, easier-to-read version of the raw score, centred on the threshold "
            "— it is not a calibrated probability."
        )

        with st.expander("What pushed this decision? (top contributing words and factors)"):
            try:
                prep = bundle["model"].named_steps["prep"]
                clf = bundle["model"].named_steps["clf"]
                feat_names = prep.get_feature_names_out()
                vec = prep.transform(build_features(row))
                contrib = np.asarray(vec.todense()).ravel() * clf.coef_.ravel()
                order = np.argsort(contrib)
                top_pos = [(feat_names[i].split("__", 1)[-1], contrib[i]) for i in order[::-1][:8] if contrib[i] > 0]
                top_neg = [(feat_names[i].split("__", 1)[-1], contrib[i]) for i in order[:8] if contrib[i] < 0]
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Pushed towards FAKE**")
                    for name, val in top_pos:
                        st.write(f"`{name}`  (+{val:.2f})")
                    if not top_pos:
                        st.caption("No feature pushed strongly towards fake.")
                with cols[1]:
                    st.markdown("**Pushed towards REAL**")
                    for name, val in top_neg:
                        st.write(f"`{name}`  ({val:.2f})")
                    if not top_neg:
                        st.caption("No feature pushed strongly towards real.")
            except Exception:
                st.caption("Explanation is only available for the Logistic Regression / Linear SVM model.")

st.divider()
with st.expander("About this model"):
    st.markdown(
        f"""
- **Model:** {bundle['name']}, trained on 14,079 job postings (EMSCAD dataset), evaluated on a held-out
  test set of 3,520 postings never seen during training.
- **Test performance:** precision 0.82, recall 0.68, F1 0.74 for the fake class (PR-AUC 0.78; a
  "always real" baseline scores 0 on all three because only ~5% of postings are fake).
- **Important limitation:** the model is noticeably weaker on entirely new scam campaigns / companies it
  has never seen (PR-AUC drops to ~0.53 in that stress test). Treat this as a **triage tool that flags ads
  for human review**, not an automatic filter.
- Full methodology, evaluation and this limitation are documented in the accompanying notebook.
"""
    )
