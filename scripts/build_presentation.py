"""Build the deck (PowerPoint) — slides + speaker-note script in one place.

Audience: Dr. Raftopoulos — a surgeon who does not code. Plain language, mechanistic.
Slides double as candidate manuscript figures. Speaker notes are the script, so every
decision can be OWNED, not attributed to a tool.

Incorporates Ioanna's 2026-07-13 review:
  * attrition slide BEFORE the metrics (sets up why yrs 5-6 fail)
  * the 15 preoperative fields named explicitly
  * RF vs GB head-to-head (answers "why not GB at yr3-4?")
  * AUC reported alongside R²
  * RACE added to demographics + sensitivity  -> the 98% finding
  * n-per-cell and the 10.5% preop threshold on the trajectory figure

Run:
    PYTHONPATH=. python scripts/build_presentation.py
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BASELINE_FEATURES

FIG = Path("presentation/figures")
OUT = Path("presentation/Bariatric_CDS_Raftopoulos_2026-07-13.pptx")

NAVY = RGBColor(0x1F, 0x35, 0x4E)
GREY = RGBColor(0x54, 0x6E, 0x7A)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREEN = RGBColor(0x1E, 0x8A, 0x4C)
W, H = Inches(13.333), Inches(7.5)


def _title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.24), Inches(12.5), Inches(0.72))
    p = tb.text_frame.paragraphs[0]
    p.text = text; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = NAVY
    if sub:
        tb2 = s.shapes.add_textbox(Inches(0.45), Inches(0.94), Inches(12.5), Inches(0.4))
        p2 = tb2.text_frame.paragraphs[0]
        p2.text = sub; p2.font.size = Pt(13); p2.font.color.rgb = GREY; p2.font.italic = True


def _bul(s, items, x=0.45, y=1.42, w=12.5, h=4.0, size=14):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        text, bold, color = (it if isinstance(it, tuple) else (it, False, None))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text; p.font.size = Pt(size); p.font.bold = bold
        p.font.color.rgb = color or RGBColor(0x21, 0x21, 0x21)
        p.space_after = Pt(6)


def _pic(s, name, x, y, w):
    f = FIG / name
    if f.exists():
        s.shapes.add_picture(str(f), Inches(x), Inches(y), width=Inches(w))


def _notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def build():
    prs = Presentation(); prs.slide_width, prs.slide_height = W, H
    B = prs.slide_layouts[6]

    # 1 ── Title
    s = prs.slides.add_slide(B)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(1.9), Inches(11.7), Inches(1.4))
    p = tb.text_frame.paragraphs[0]
    p.text = "Predicting Weight-Loss Trajectories After Bariatric Surgery"
    p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = NAVY
    tb2 = s.shapes.add_textbox(Inches(0.8), Inches(3.2), Inches(11.7), Inches(0.7))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = "Flagging the patients likely to regain — before they do"
    p2.font.size = Pt(18); p2.font.color.rgb = GREY
    _bul(s, [("786 patients  •  Random Forest  •  5 phenotypes  •  a working surgeon-facing app",
              True, NAVY),
             "Kaushik Perkari  |  Manuscript 1B (follow-up to Ioanna's 1A)  |  July 2026"],
         y=4.25, size=15)
    _notes(s, """SAY: A patient asks "how much will I lose, and will I keep it off?" Today the honest
answer is a population average. But patients vary enormously — and by the time we SEE regain,
we've missed the window to act.

This tool does three things:
  1. Forecasts for THIS patient, not the average patient
  2. Tells you how much to trust it — and REFUSES to answer when it doesn't know
  3. Sorts patients into trajectory types, to catch likely regainers early

CLOSE THE INTRO WITH THIS — it buys credibility and defuses the honesty slide later:
"I'll also show you a problem I found in my own analysis. I'd rather you hear it from me than
find it yourselves." """)

    # 2 ── Attrition (Ioanna: BEFORE the metrics)
    s = prs.slides.add_slide(B)
    _title(s, "First — who do we actually still have?",
           "Read this before any performance number; it explains everything that follows")
    _pic(s, "fig0b_sankey.png", 0.35, 1.55, 12.6)
    _notes(s, """IOANNA EXPLICITLY ASKED FOR THIS SLIDE, AND FOR IT TO COME FIRST.

SAY: Before I show you a single accuracy number, look at who we still have.

We start with 786 patients. By year 1, 593 have data. By year 2, 386. By year 5, only 85 —
about 10% of the cohort. People stop coming to clinic.

This single chart explains almost everything that follows: why years 2-4 are strong, and why
years 5 and 6 are hopeless. It is not that the models are bad. It is that there is almost
nothing left to learn from.

WHY THIS MATTERS POLITICALLY: reviewers attacked Ioanna's models as "poorly implemented"
while completely missing that she had openly disclosed 90% attrition. So we put it first and
make it impossible to miss.

THE ATTRITION IS PROTECTIVE, NOT INFLATIONARY (Ioanna's new sensitivity analysis, Table S9):
the patients who REMAIN at years 5-6 are disproportionately Hispanic (SMD -0.43), started with
LOWER preop weight loss (SMD -0.35) and higher fat mass. Lower preop weight loss is the WORSE
prognostic group. So if anything the late-year trajectories are conservatively biased DOWNWARD,
not inflated. Say this if anyone claims the late results are cherry-picked: "the opposite — the
sicker, lower-preop-loss patients are the ones who stayed, so these numbers understate, not
overstate, the benefit." """)

    # 3 ── Pipeline + the 15 fields named
    s = prs.slides.add_slide(B)
    _title(s, "How it works — and what each piece of code does")
    _pic(s, "fig1_pipeline.png", 0.3, 1.35, 12.7)
    feats = ", ".join(BASELINE_FEATURES)
    _bul(s, [
        ("THE 15 PREOPERATIVE FIELDS THE MODEL SEES:", True, NAVY),
        (feats, False, None),
        ("data_load.py cleans and removes 16 duplicate rows (802 → 786).  preprocess.py turns each "
         "patient into those 15 numbers.  reproduce_models.py trains one Random Forest per year — "
         "ONCE, then saved.  predict.py + reliability.py produce the forecast and its trust rating.  "
         "phenotype.py groups patients.  streamlit_app.py is what you click on.", False, None),
    ], y=4.35, size=12)
    _notes(s, """IOANNA: "When you say 15 preoperative numbers, you have to name exactly what these
fields are." So read them out: age, sex, race, height, initial BMI, initial weight, initial BMR,
visceral fat, body-fat %, fat mass, fat-free mass, time to surgery, surgery type, preop BMI,
preop total body weight loss.

WALK THE BOXES LEFT TO RIGHT.

CLEAN + DEDUP — 802 ROWS but only 786 PATIENTS. Sixteen duplicate rows. Ioanna explained the
cause: the curator assigned the SAME ID to two genuinely different people (you'd see "ID 2,
African-American male" and "ID 2, white female"), and merging on ID cross-multiplied them.
NOTE: she deliberately KEPT all 802 in manuscript 1A. We remove them. This is an internal
discussion — NOT for the manuscript.

TRAINED ONCE, THEN SERVED — we do not retrain per patient. That's why the app is instant.""")

    # 4 ── What a Random Forest is + 5-model comparison
    s = prs.slides.add_slide(B)
    _title(s, "The models — what a Random Forest actually is",
           "Hundreds of simple decision trees, voting")
    _pic(s, "fig2_model_comparison.png", 0.3, 1.5, 8.6)
    _bul(s, [
        ("THE ANALOGY", True, NAVY),
        "Ask 300 junior doctors to guess a patient's weight loss. Each is deliberately simple — "
        "they may only ask a few yes/no questions (Is BMI over 45? Did they lose 10% preop?). "
        "Each guess is mediocre. Average all 300 and the errors cancel out.",
        "That is a Random Forest. Each 'junior doctor' is a decision tree.",
        ("TWO SAFEGUARDS", True, NAVY),
        "No time travel: the year-2 model sees baseline + year-1 only. Never year 2 or later. "
        "I verified this in every saved model's feature list.",
        "Reproducible: every model and split is seeded (random_state = 42).",
    ], x=9.0, y=1.55, w=4.1, h=5.4, size=11)
    _notes(s, """DEFINE IT FIRST — DO NOT SKIP THE ANALOGY. Ioanna's warning: "However many times I
have explained random forest / gradient boosting / MLP / SVR to Dr. R, he doesn't remember."
So keep it concrete: 300 simple doctors voting, errors cancel out.

LEAKAGE ("no time travel"): if a year-2 model saw the year-2 answer it would look brilliant and
be useless. This is the single most common way a medical ML paper turns out wrong — and it is
the bug Ioanna caught in an earlier version of HER code. She confirmed OUR pipeline does not
have it. I verified by listing every saved model's inputs.

random_state = 42 — just a fixed seed so re-running gives identical numbers.""")

    # 5 ── RF vs GB (NEW — the reviewer question)
    s = prs.slides.add_slide(B)
    _title(s, "Why Random Forest and not Gradient Boosting?",
           "The question a reviewer will ask — answered with our own data, not the literature's")
    _pic(s, "fig2b_rf_vs_gb.png", 0.5, 1.5, 12.3)
    _bul(s, [
        ("Honest answer: Gradient Boosting DOES beat Random Forest at year 3 (R² 0.70 vs 0.61), and "
         "year 4 is a dead tie. But at year 5 Gradient Boosting collapses (R² = −3.8) where Random "
         "Forest degrades gracefully. Random Forest also discriminates better overall (mean AUC 0.80 "
         "vs 0.75).", True, RED),
        ("And the clustering forces the choice: a reviewer correctly told Ioanna you cannot switch "
         "models between years for k-means — it imports nonlinearities. One model must serve all "
         "years. So we choose for CONSISTENCY, and Gradient Boosting's year-5 collapse disqualifies it.",
         True, GREEN),
    ], y=5.5, size=12)
    _notes(s, """IOANNA ASKED FOR THIS SLIDE SPECIFICALLY. She said a reviewer will ask: "why did you
pick random forest and not gradient boosting for years 3 and 4?" — because her S5 table shows GB
winning there. Have the answer ready.

OUR OWN NUMBERS (5-fold CV, deduped data, same features for both):
  yr1  RF 0.249  GB 0.228   -> RF
  yr2  RF 0.513  GB 0.452   -> RF
  yr3  RF 0.608  GB 0.698   -> GB WINS. SAY SO. Don't hide it.
  yr4  RF 0.725  GB 0.723   -> tie
  yr5  RF -0.346 GB -3.804  -> GB CATASTROPHICALLY COLLAPSES
  AUC mean: RF 0.795, GB 0.748

THE ARGUMENT: you must pick ONE model for all years (the reviewer's own logic about k-means
clustering — you cannot mix orthogonal models across years). So you optimise for consistency,
not for a single year's peak. GB's year-5 collapse rules it out.

AUC POINT — Ioanna gets excited about this: our AUCs are 0.90 / 0.91 / 0.95 at years 2/3/4.
She said anything above 0.8 with real-world data is "amazing" and that honest reviewers would
commend us for it.""")

    # 6 ── Reliability + AUC
    s = prs.slides.add_slide(B)
    _title(s, "The tool refuses to guess when it cannot predict",
           "Arguably the most clinically important design decision")
    _pic(s, "fig3_reliability.png", 0.6, 1.6, 12.1)
    _bul(s, [
        ("R² = of all the variation between patients, how much does the model explain? "
         "AUC = can it correctly rank who loses more? Thresholds fixed IN ADVANCE: green ≥ 0.40, "
         "amber 0.20–0.40, red < 0.20.", False, None),
        ("Years 2–4: trustworthy. Year 1: rough guide. Years 5–6: we show NOTHING rather than a "
         "number you might act on.", True, RED),
    ], y=5.35, size=12.5)
    _notes(s, """DEFINE R-SQUARED PLAINLY: "Of all the variation between patients, how much does the
model actually explain? Zero = it knows nothing. In medicine, above 0.4 is genuinely useful."

IOANNA'S FRAMING FOR THE MISSING HALF — use this, it's excellent:
"Our best is about 50% explained variance at year 2. What is the other 50%? Factors we never
measured. We have psychometric scores only at the FIRST visit — nothing longitudinal. A patient
who was abused as a child and eats for comfort; the psychological support they get after joining
the program. Dr. R's team almost certainly HAS these scores — they're sitting in scanned notes in
the EMR that nobody has entered. So the missing 50% is not mysterious; it is unmeasured."

AUC ADDED AT IOANNA'S REQUEST. She wants both metrics because R² is the stringent one but AUC is
what a clinician can use. Ours: 0.90 / 0.91 / 0.95 at years 2/3/4.

THE HONEST COST OF CHOOSING RANDOM FOREST: year 5 used to be amber because SVR scored 0.26 there.
RF scores 0.064. So year 5 correctly turned RED. We LOST a number — but it was one nobody should
have trusted.""")

    # 7 ── k derivation
    s = prs.slides.add_slide(B)
    _title(s, "The number of groups was measured, not chosen",
           "k swept from 2 to 10 and scored — this is NOT hard-coding")
    _pic(s, "fig4_silhouette.png", 0.35, 1.55, 6.4)
    _pic(s, "fig5_umap.png", 7.05, 1.55, 5.1)
    _bul(s, [
        ("Clustering = the computer groups similar patients without being told what to look for. "
         "Silhouette = are the groups genuinely tight and separated? UMAP = a 2-D map so you can see "
         "them (a visual aid, not the analysis).", False, None),
        ("k=4 scores 0.7165 and k=5 scores 0.7163 — a tie to four decimals. We report 5: it matches "
         "1A and the split is clinically interpretable. Documented as a tie-break in the code.", True, RED),
    ], y=6.0, size=12),
    _notes(s, """IOANNA RESOLVED YOUR BIGGEST WORRY HERE. Her words:
"If you only set the selection of clusters between two and ten, that's NOT hard coding. If you
said 'I want five clusters', THAT is hard coding. You didn't hard code — you wouldn't have a chart
of silhouette scores if you had."

THE SILHOUETTE CHART IS YOUR PROOF. Show it and the accusation dies.

BE HONEST ABOUT THE TIE: k=4 scores 0.7165, k=5 scores 0.7163. There is no meaningful difference.
We report 5 because it matches 1A and is clinically interpretable — and it's documented in the code.

DO NOT claim "5 is clearly optimal." The figure disproves it and someone will notice.

Ioanna's own tip: the curve must come back DOWN inside your range. If it were still rising at k=10
you'd have chosen too small a range. Ours peaks and falls — the range was right.""")

    # 8 ── Phenotypes (with race)
    s = prs.slides.add_slide(B)
    _title(s, "The five phenotypes", "Ordered by preoperative weight loss — with race, and n per point")
    _pic(s, "fig6_trajectories.png", 0.25, 1.45, 6.7)
    _pic(s, "fig7_demographics.png", 7.1, 1.45, 5.9)
    _notes(s, """TWO THINGS IOANNA ASKED FOR AND YOU NOW HAVE:
  1. The 10.5% preop threshold drawn, with phenotypes below it DASHED. Phenotype 1 (8.8% preop) is
     the one below the threshold — the group we already know does worse long-term.
  2. n shown at every point, because she challenged two numbers:
       - Phenotype 1 RISES at year 6 (22.9% -> 32.2%)
       - Phenotype 3 shows ~44% loss at year 6 — she said "who are these people? I didn't have such
         results."
     Both are tiny-sample artifacts. The n labels and the shaded "sparse" zone make that visible.
     If challenged: "those late points rest on a handful of patients; that's why years 5-6 are red."

VALIDATION POINT WORTH MAKING: ordered by preop weight loss I get 8.8 / 11.0 / 11.5 / 12.3 / 12.9%.
Ioanna's Supplementary Table 7 reports 9.0 / 10.7 / 11.2 / 12.7 / 12.8. Two independent pipelines,
essentially the same answer.

NOW LOOK AT THE RACE ROWS ON THE HEATMAP — and go straight to the next slide before anyone else
gets there.""")

    # 9 ── The finding (corrected after reconciling with 1A Table 6)
    s = prs.slides.add_slide(B)
    _title(s, "Finding: phenotypes are structured by procedure and sex — not race",
           "Checked against Ioanna's Table 6 and Table 2, as she asked. Procedure + sex are the drivers; race is a secondary correlate.")
    _pic(s, "fig8_sensitivity.png", 0.35, 1.5, 12.6)
    _bul(s, [
        ("71% of phenotype membership is predictable from procedure + sex alone — the real structural "
         "drivers. In 1A's own clusters (Table 6), the two female-sleeve groups differ by preop weight "
         "loss, NOT race (65% vs 57% Hispanic).", True, RED),
        ("Race is a secondary correlate: it tracks preop and 1-year weight loss (Table 2, p<0.001) but "
         "NOT 4-year outcome (p=0.15). A leaner re-clustering can over-separate by race — we flag that "
         "as feature-sensitivity, not a stable biological signal.", False, None),
        ("Clustering on trajectory SHAPE alone drops predictability to 44% and reveals a real "
         "poor→strong responder gradient — including a strong-early-loss group that quietly regains by "
         "year 4: the 'Quiet Regainer', now a high-alert flag in the app.", True, GREEN),
    ], y=5.4, size=11),
    _notes(s, """*** THE MOST IMPORTANT SLIDE — AND I CORRECTED IT AFTER CHECKING IOANNA'S TABLES. ***

WHAT CHANGED: My first pass said the phenotypes were "98% procedure + sex + RACE" and called them
Hispanic-vs-White groups. Ioanna asked me to double-check against her Table 6 (cluster demographics)
and Table 2 (stratification). I did — and she is right: RACE IS NOT A PRIMARY DRIVER.

THE EVIDENCE:
  - Her primary clustering (1A) uses BASELINE PREOP FEATURES ONLY. Its female-sleeve clusters (her
    2 and 3) split by preop TBWL (10.7% vs 11.2%), and are only mildly race-skewed (65% vs 57%
    Hispanic). Not a race split.
  - My re-clustering used a leaner feature set (15 preop features + trajectories), which let the
    race dummies dominate and produced an artificial 91%-Hispanic vs 98%-White split. That is a
    feature-weighting artifact, not a finding. Adding more trajectory features didn't fix it;
    it's driven by the small feature panel.
  - Table 2 stratification: race is significant for preop and 1-yr TBWL (p<0.001) but NOT 4-yr
    (p=0.15). So race correlates with the STARTING point, not the long-term trajectory.

SO THE HONEST, 1A-CONSISTENT FINDING (say this):
"The phenotypes are structured mainly by procedure and sex — 71% of membership is predictable from
those two alone. Race correlates with preoperative and early weight loss, but it does not define the
clusters and it washes out of the 4-year outcome. When we strip demographics out and cluster on the
weight-loss curve itself, predictability falls to 44% and a genuine responder gradient appears."

THE QUIET REGAINER (Ioanna called this a clinical goldmine): in the trajectory-only clustering, one
group loses strongly early then quietly regains by year 4 — the patient you'd never flag from their
chart. It is now a high-alert flag in the app.

DON'T over-claim race. If asked about the Hispanic/White thing from an earlier draft: "That was an
artifact of a reduced-feature re-clustering; against 1A's full clustering it doesn't hold, and I
corrected it."

DECISION (unchanged): 1A-aligned clustering stays primary for comparability; trajectory-shape is the
sensitivity analysis; the procedure/sex structuring is stated plainly in the Results.""")

    # 10 ── Mechanism + next
    s = prs.slides.add_slide(B)
    _title(s, "Where a prediction comes from — and what's next")
    _bul(s, [
        ("IT IS NOT AN EXTRAPOLATION. IT IS NOT THE CLUSTER AVERAGE.", True, RED),
        "We take this patient's 15 preoperative values, run them through the Random Forest for that "
        "specific year, and hundreds of trees vote. The output is that patient's own predicted TBWL%.",
        "Order matters: we PREDICT first, then assign the phenotype from that prediction. The cluster "
        "does not create the number — the number places the patient in a cluster.",
        ("STATUS vs 1A", True, NAVY),
        "Reproduced her Table S5 performance; adopted her conventions; independently reproduced her "
        "5 clusters and preop-TBWL ordering (8.8/11.0/11.5/12.3/12.9 vs her 9.0/10.7/11.2/12.7/12.8).",
        "Found 3 real bugs in her shared scripts — she confirmed all three.",
        ("NEXT", True, NAVY),
        "Decide primary vs sensitivity clustering → finalise figures → draft the manuscript. The "
        "STREAMLIT APP is the novel contribution; the modelling is supporting material. Target: Obesity Surgery.",
    ], y=1.45, size=13)
    _notes(s, """KILL THE TWO MISREADS. Both would make the tool sound worthless:
  NOT extrapolation — we are not drawing a line forward.
  NOT the cluster average — we are not quoting what the group typically does.
It is a PER-PATIENT model output. And we PREDICT FIRST, CLUSTER SECOND.

IOANNA'S STRATEGIC POINT — the most important thing she said about the manuscript:
"Your major figures are going to be the Streamlit app. We don't want to present the same kind of
data again. Everything you did is fine for supporting documentation, but now you're presenting your
Streamlit app, which is amazing."
=> THE TOOL IS THE PAPER. The modelling and clustering become supplementary.

NEXT WEEK (her list): repeat with gradient boosting [DONE], add race to clusters [DONE], add AUC
[DONE], attrition figure [DONE] — then START WRITING: methods referencing 1A, results with RF + GB,
supplementary verification. She also wants a walkthrough of how you built this with Claude.

DO NOT attribute any of this to a tool. Every decision here is one you can defend: why Random
Forest, why 42, why the red zone, why 5 clusters, and why the phenotypes may be demographic strata.""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Deck written: {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
