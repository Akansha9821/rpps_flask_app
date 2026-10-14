
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Load CSV files
projects = pd.read_csv("project.csv")
tasks = pd.read_csv("task.csv")

# 2. Clean column names
projects.columns = projects.columns.str.strip().str.lower()
tasks.columns = tasks.columns.str.strip().str.lower()

project_text_columns = [
    "project_title",
    "candidate_level",
    "subject_category",
    "difficulty_level",
    "primary_skills",
    "tools_technologies",
    "project_description",
]

task_text_columns = [
    "task_title",
    "task_description",
    "task_category",
    "candidate_level",
    "required_skills",
    "tools_technologies",
    "acceptance_criteria",
]

# 3. Handle missing columns and values
for col in project_text_columns:
    if col not in projects.columns:
        projects[col] = ""
    projects[col] = projects[col].fillna("").astype(str)

for col in task_text_columns:
    if col not in tasks.columns:
        tasks[col] = ""
    tasks[col] = tasks[col].fillna("").astype(str)

# 4. Combine tasks for each project
tasks["task_text"] = tasks[task_text_columns].agg(
    " ".join, axis=1
)

task_summary = (
    tasks.groupby("project_id")["task_text"]
    .apply(lambda values: " ".join(values))
    .reset_index()
    .rename(columns={"task_text": "project_task_text"})
)

projects = projects.merge(
    task_summary, on="project_id", how="left"
)

projects["project_task_text"] = (
    projects["project_task_text"].fillna("")
)

# 5. Combine project and task information
projects["combined_text"] = (
    projects[project_text_columns].agg(" ".join, axis=1)
    + " "
    + projects["project_task_text"]
)

# 6. Build TF-IDF vectors once
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

project_vectors = vectorizer.fit_transform(
    projects["combined_text"]
)


# 7. Recommend projects from user-entered skills
def recommend_projects(candidate_skills,
                      candidate_level=None,
                      top_n=5):

    if not candidate_skills.strip():
        return pd.DataFrame()

    candidate_vector = vectorizer.transform(
        [candidate_skills]
    )

    scores = cosine_similarity(
        candidate_vector, project_vectors
    ).flatten()

    results = projects.copy()
    results["similarity_score"] = scores

    if candidate_level:
        level_match = (
            results["candidate_level"].str.strip().str.lower()
            == candidate_level.strip().lower()
        )

        results["match_score"] = (
            results["similarity_score"] * 0.85
            + level_match.astype(float) * 0.15
        )
    else:
        results["match_score"] = results["similarity_score"]

    results["match_percentage"] = (
        results["match_score"] * 100
    ).round(2)

    output_columns = [
        "project_id",
        "project_title",
        "candidate_level",
        "subject_category",
        "primary_skills",
        "tools_technologies",
        "match_percentage",
    ]

    return (
        results.sort_values("match_score", ascending=False)
        [output_columns]
        .head(top_n)
    )


# 8. Fetch tasks for the selected project
def get_project_tasks(project_id):
    project_tasks = tasks[
        tasks["project_id"] == project_id
    ].copy()

    if "task_sequence" in project_tasks.columns:
        project_tasks = project_tasks.sort_values(
            "task_sequence"
        )

    columns = [
        col for col in [
            "task_title",
            "task_description",
            "required_skills",
            "tools_technologies",
            "estimated_duration_days",
            "task_priority",
            "task_status",
        ]
        if col in project_tasks.columns
    ]

    return project_tasks[columns].to_dict("records")


# 9. Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    selected_skills = ""
    selected_level = ""
    error = None

    if request.method == "POST":
        selected_skills = request.form.get(
            "skills", ""
        ).strip()

        selected_level = request.form.get(
            "candidate_level", ""
        ).strip()

        if not selected_skills:
            error = "Please enter at least one skill."
        else:
            results = recommend_projects(
                candidate_skills=selected_skills,
                candidate_level=selected_level or None,
                top_n=5,
            )

            recommendations = results.to_dict(
                orient="records"
            )

            for project in recommendations:
                project["tasks"] = get_project_tasks(
                    project["project_id"]
                )

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_skills=selected_skills,
        selected_level=selected_level,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
