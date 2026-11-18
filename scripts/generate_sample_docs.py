"""Generate non-sensitive, educational medical reference PDF and DOCX sample documents."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.shared import Pt


def _write_pdf_pages(output_path: Path, header: str, pages: list[list[tuple[str, str]]]) -> None:
    """Create a clean multi-page text-searchable PDF using PyMuPDF Story/insert_text."""
    doc = fitz.open()
    total_pages = len(pages)
    for page_idx, blocks in enumerate(pages, start=1):
        page = doc.new_page(width=612, height=792)  # US Letter
        y = 48.0
        # Repeated header (intentionally tests repeated-header cleaning)
        page.insert_text((50, y), header, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
        y += 28.0

        for kind, content in blocks:
            if kind == "h1":
                page.insert_text((50, y), content, fontsize=15, fontname="helv", color=(0.05, 0.25, 0.45))
                y += 24.0
            elif kind == "h2":
                y += 4.0
                page.insert_text((50, y), content, fontsize=12, fontname="helv", color=(0.08, 0.35, 0.55))
                y += 18.0
            elif kind == "p":
                rect = fitz.Rect(50, y, 562, 740)
                rc = page.insert_textbox(
                    rect,
                    content,
                    fontsize=10.5,
                    fontname="helv",
                    lineheight=1.35,
                    color=(0.1, 0.1, 0.1),
                )
                # Estimate height used by line count
                approx_lines = max(1, (len(content) // 82) + content.count("\n") + 1)
                y += approx_lines * 14.5 + 10.0
                if rc < 0:
                    y += 20.0
            elif kind == "table":
                # Draw pipe-formatted lines cleanly inside a monospace block
                for line in content.splitlines():
                    page.insert_text((50, y), line, fontsize=9.2, fontname="cour", color=(0.1, 0.1, 0.1))
                    y += 14.0
                y += 8.0

        # Repeated footer
        page.insert_text(
            (50, 760),
            f"Page {page_idx} of {total_pages}",
            fontsize=9,
            fontname="helv",
            color=(0.4, 0.4, 0.4),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()


def generate_hypertension_pdf(samples_dir: Path) -> Path:
    output = samples_dir / "WHO_Hypertension_Clinical_Guideline.pdf"
    header = "MediSage Educational Reference — Adult Hypertension Clinical Summary"
    pages = [
        [
            ("h1", "Adult Hypertension: Classification and Diagnostic Thresholds"),
            ("h2", "1. Blood Pressure Classification Overview"),
            (
                "p",
                "Systemic arterial hypertension in adults is defined by persistently elevated office "
                "blood pressure readings obtained across at least two separate clinical visits. Standard "
                "measurement requires the patient to rest quietly for 5 minutes seated with the arm "
                "supported at heart level and an appropriately sized cuff.",
            ),
            ("h2", "Blood Pressure Categories Table"),
            (
                "table",
                "| Category | Systolic (mmHg) | Diastolic (mmHg) | Clinical Action |\n"
                "| Normal | < 120 | and < 80 | Re-evaluate annually |\n"
                "| Elevated | 120 - 129 | and < 80 | Lifestyle modification, recheck 3-6 months |\n"
                "| Stage 1 Hypertension | 130 - 139 | or 80 - 89 | Risk assessment and lifestyle/drug review |\n"
                "| Stage 2 Hypertension | >= 140 | or >= 90 | Confirm and initiate combination/pharmacotherapy |\n"
                "| Hypertensive Crisis | > 180 | and/or > 120 | Urgent evaluation for target-organ injury |",
            ),
            (
                "p",
                "Out-of-office confirmation using home blood pressure monitoring (HBPM) or 24-hour "
                "ambulatory blood pressure monitoring (ABPM) is recommended to distinguish true "
                "hypertension from white-coat or masked hypertension.",
            ),
        ],
        [
            ("h1", "Cardiovascular Risk and Non-Pharmacological Interventions"),
            ("h2", "2. Core Lifestyle Modifications"),
            (
                "p",
                "Evidence-based non-pharmacological interventions lower systolic blood pressure and enhance "
                "antihypertensive medication efficacy across all stages of hypertension. Every adult with "
                "elevated or hypertensive blood pressure should receive structured lifestyle counseling.",
            ),
            (
                "p",
                "Dietary Sodium Reduction: Limiting dietary sodium intake to less than 2 grams per day "
                "(approximately 5 grams of dietary salt) typically reduces systolic blood pressure by 4 to 6 mmHg. "
                "Dietary Approaches to Stop Hypertension (DASH) eating patterns emphasizing vegetables, fruits, "
                "whole grains, low-fat dairy, and potassium-rich foods produce systolic reductions of 8 to 11 mmHg.",
            ),
            (
                "p",
                "Physical Activity and Weight Management: Regular moderate-intensity aerobic exercise of at "
                "least 150 minutes per week (such as brisk walking or cycling) lowers systolic blood pressure "
                "by 5 to 8 mmHg. Weight loss of roughly 1 mmHg per kilogram of excess body weight lost, "
                "smoking cessation, and alcohol moderation further reduce long-term cardiovascular risk.",
            ),
        ],
        [
            ("h1", "Pharmacological Therapy Reference and Contraindications"),
            ("h2", "3. First-Line Antihypertensive Drug Classes"),
            (
                "p",
                "When pharmacological treatment is indicated by a clinician, initial therapy is typically "
                "selected from four major evidence-based drug classes: (1) thiazide or thiazide-like diuretics "
                "(such as chlorthalidone or hydrochlorothiazide), (2) angiotensin-converting enzyme (ACE) "
                "inhibitors (such as lisinopril or enalapril), (3) angiotensin II receptor blockers (ARBs "
                "such as losartan or valsartan), and (4) dihydropyridine calcium channel blockers (CCBs such "
                "as amlodipine).",
            ),
            ("h2", "4. Pregnancy and Key Safety Contraindications"),
            (
                "p",
                "ACE inhibitors, ARBs, and direct renin inhibitors are strictly contraindicated in pregnancy "
                "due to fetal renal toxicity and teratogenic risk. Preferred agents for hypertension during "
                "pregnancy include labetalol, extended-release nifedipine, or methyldopa under specialist "
                "obstetric supervision. Combining an ACE inhibitor with an ARB simultaneously is not recommended "
                "because dual renin-angiotensin blockade increases hyperkalemia and acute kidney injury risk.",
            ),
        ],
        [
            ("h1", "Treatment Targets, Follow-Up, and Urgent Escalation"),
            ("h2", "5. Blood Pressure Targets and Monitoring"),
            (
                "p",
                "For most adults receiving antihypertensive therapy, a general treatment target below 140/90 mmHg "
                "is recommended, with a tighter target below 130/80 mmHg commonly pursued in individuals with "
                "established cardiovascular disease, chronic kidney disease, or diabetes when tolerated without "
                "dizziness or symptomatic hypotension.",
            ),
            ("h2", "6. Red-Flag Symptoms Requiring Urgent Care"),
            (
                "p",
                "Patients with blood pressure above 180/120 mmHg accompanied by acute chest pain, shortness of "
                "breath, neurological deficits, severe headache, visual changes, or acute kidney injury signs "
                "require immediate emergency department evaluation for hypertensive emergency.",
            ),
        ],
    ]
    _write_pdf_pages(output, header, pages)
    return output


def generate_diabetes_pdf(samples_dir: Path) -> Path:
    output = samples_dir / "Type2_Diabetes_Management_Reference.pdf"
    header = "MediSage Educational Reference — Type 2 Diabetes Clinical Guide"
    pages = [
        [
            ("h1", "Type 2 Diabetes: Screening and Diagnostic Criteria"),
            ("h2", "1. Laboratory Diagnostic Thresholds"),
            (
                "p",
                "Type 2 diabetes mellitus is characterized by progressive insulin resistance and relative "
                "beta-cell insulin deficiency. Unless unequivocal symptomatic hyperglycemia is present, "
                "diagnosis requires two abnormal test results from the same sample or two separate test days.",
            ),
            ("h2", "Diagnostic Reference Table"),
            (
                "table",
                "| Glycemic Test | Normal | Prediabetes | Diabetes Mellitus |\n"
                "| HbA1c (%) | < 5.7% | 5.7% - 6.4% | >= 6.5% |\n"
                "| Fasting Plasma Glucose (mg/dL) | < 100 | 100 - 125 | >= 126 |\n"
                "| 2-Hour 75g OGTT (mg/dL) | < 140 | 140 - 199 | >= 200 |\n"
                "| Random Plasma Glucose | N/A | N/A | >= 200 with symptoms |",
            ),
            (
                "p",
                "Screening is generally considered in asymptomatic adults aged 35 years and older, or earlier "
                "in adults with overweight or obesity plus additional risk factors such as family history, "
                "prior gestational diabetes, hypertension, or dyslipidemia.",
            ),
        ],
        [
            ("h1", "Glycemic Targets and Foundational Care"),
            ("h2", "2. Individualized HbA1c Goals"),
            (
                "p",
                "An HbA1c goal below 7.0% (53 mmol/mol) is appropriate for many non-pregnant adults with "
                "Type 2 diabetes when achievable safely without significant hypoglycemia. Less stringent targets "
                "such as below 8.0% may be appropriate for patients with limited life expectancy, extensive "
                "comorbidities, or severe hypoglycemia history.",
            ),
            (
                "p",
                "Foundational care combines diabetes self-management education, individualized nutrition "
                "emphasizing fiber-rich non-starchy vegetables and reduced refined carbohydrates, and at least "
                "150 minutes per week of moderate aerobic exercise paired with resistance training twice weekly.",
            ),
        ],
        [
            ("h1", "First-Line and Organ-Protective Pharmacotherapy"),
            ("h2", "3. Metformin and Renal Considerations"),
            (
                "p",
                "Metformin remains a widely used first-line oral agent for Type 2 diabetes due to its glucose-lowering "
                "efficacy, neutral-to-favorable weight profile, and low hypoglycemia risk when used as monotherapy. "
                "Metformin is contraindicated when estimated glomerular filtration rate (eGFR) falls below "
                "30 mL/min/1.73 m2 because of lactic acidosis risk, and initiation is generally avoided between "
                "30 and 45 mL/min/1.73 m2. Long-term metformin therapy requires periodic vitamin B12 monitoring.",
            ),
            ("h2", "4. Cardiorenal Agents: SGLT2 Inhibitors and GLP-1 Receptor Agonists"),
            (
                "p",
                "In patients with Type 2 diabetes and established heart failure or chronic kidney disease with "
                "albuminuria, sodium-glucose cotransporter-2 (SGLT2) inhibitors provide proven hospitalization "
                "and renal progression benefits. In patients with established atherosclerotic cardiovascular disease "
                "(ASCVD), glucagon-like peptide-1 (GLP-1) receptor agonists or SGLT2 inhibitors with demonstrated "
                "cardiovascular benefit are prioritized alongside foundational therapy.",
            ),
        ],
        [
            ("h1", "Hypoglycemia Recognition and Acute Emergencies"),
            ("h2", "5. Hypoglycemia Classification and the 15-15 Rule"),
            (
                "p",
                "Level 1 hypoglycemia is defined as blood glucose below 70 mg/dL (3.9 mmol/L) and >= 54 mg/dL, "
                "while Level 2 clinically significant hypoglycemia is below 54 mg/dL (3.0 mmol/L). Level 3 is a "
                "severe event characterized by altered mental or physical status requiring third-party assistance.",
            ),
            (
                "p",
                "For conscious individuals with blood glucose below 70 mg/dL, the standard 15-15 rule recommends "
                "consuming 15 grams of fast-acting carbohydrates (such as glucose tablets or 4 oz fruit juice), "
                "waiting 15 minutes, and rechecking blood glucose; repeat if still below 70 mg/dL. Severe "
                "hypoglycemia with unresponsiveness requires emergency glucagon administration and urgent medical aid.",
            ),
        ],
    ]
    _write_pdf_pages(output, header, pages)
    return output


def generate_respiratory_docx(samples_dir: Path) -> Path:
    output = samples_dir / "Acute_Respiratory_Infection_Stewardship.docx"
    doc = DocxDocument()

    doc.add_heading("Acute Respiratory Infections: Antibiotic Stewardship & Severity Guide", level=1)
    doc.add_heading("1. Distinguishing Viral URI from Bacterial Infection", level=2)
    doc.add_paragraph(
        "Most acute upper respiratory tract infections—including the common cold, viral pharyngitis, "
        "and uncomplicated acute bronchitis—are caused by respiratory viruses and resolve with supportive "
        "care alone. Routine antibiotic prescribing for uncomplicated acute bronchitis or viral rhinosinusitis "
        "provides minimal clinical benefit while increasing adverse drug events and antimicrobial resistance."
    )
    doc.add_paragraph(
        "Supportive management emphasizes hydration, rest, fever/analgesic relief where clinically appropriate, "
        "and clear return precautions. Bacterial rhinosinusitis is considered mainly when symptoms persist "
        "beyond 10 days without improvement, present with severe fever (>=39 C) and purulent facial pain for "
        "3-4 consecutive days, or follow a 'double-sickening' worsening pattern after initial viral improvement."
    )

    doc.add_page_break()

    doc.add_heading("2. Community-Acquired Pneumonia Severity: CURB-65 Criteria", level=2)
    doc.add_paragraph(
        "When community-acquired pneumonia (CAP) is clinically suspected and radiographically evaluated, "
        "the CURB-65 score assists clinicians in estimating 30-day mortality risk and determining the "
        "appropriate site of care (outpatient vs inpatient vs intensive care)."
    )

    table = doc.add_table(rows=6, cols=3)
    table.style = "Table Grid"
    headers = ["CURB-65 Criterion", "Clinical Threshold", "Points"]
    rows_data = [
        ["Confusion", "New-onset mental confusion or disorientation", "1 point"],
        ["Urea", "Blood urea nitrogen > 7 mmol/L (BUN > 19 mg/dL)", "1 point"],
        ["Respiratory Rate", "Respiratory rate >= 30 breaths per minute", "1 point"],
        ["Blood Pressure", "Systolic BP < 90 mmHg or Diastolic BP <= 60 mmHg", "1 point"],
        ["Age", "Age >= 65 years", "1 point"],
    ]
    for col_idx, text in enumerate(headers):
        run = table.rows[0].cells[col_idx].paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(10)
    for row_idx, rdata in enumerate(rows_data, start=1):
        for col_idx, val in enumerate(rdata):
            table.rows[row_idx].cells[col_idx].paragraphs[0].text = val

    doc.add_paragraph(
        "CURB-65 Risk Stratification: A score of 0 or 1 generally indicates low risk suitable for outpatient "
        "management if social support and oral intake are reliable. A score of 2 warrants short inpatient "
        "observation or closely supervised outpatient hospital-at-home care. A score of 3 to 5 indicates "
        "severe pneumonia requiring urgent inpatient hospitalization and assessment for intensive care."
    )

    doc.add_page_break()

    doc.add_heading("3. Red-Flag Respiratory Warning Signs", level=2)
    doc.add_paragraph(
        "Regardless of initial presumptive diagnosis, urgent medical evaluation is warranted for resting "
        "oxygen saturation (SpO2) below 92% on room air, rapid breathing (>=30 breaths/min), cyanosis, "
        "pleuritic chest pain with hemodynamic instability, inability to maintain oral fluids, or new "
        "confusion. Clinicians also monitor patients with chronic cardiopulmonary disease closely for "
        "decompensation during acute respiratory illnesses."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output))
    return output


def generate_all_samples(
    samples_dir: Path | None = None, *, overwrite: bool = False
) -> list[Path]:
    """Generate (or reuse existing) non-sensitive educational sample medical documents."""
    target_dir = samples_dir or (PROJECT_ROOT / "data" / "samples")
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        (target_dir / "WHO_Hypertension_Clinical_Guideline.pdf", generate_hypertension_pdf),
        (target_dir / "Type2_Diabetes_Management_Reference.pdf", generate_diabetes_pdf),
        (target_dir / "Acute_Respiratory_Infection_Stewardship.docx", generate_respiratory_docx),
    ]
    paths: list[Path] = []
    for path, builder in expected:
        if not overwrite and path.exists() and path.stat().st_size > 0:
            paths.append(path)
        else:
            paths.append(builder(target_dir))
    return paths


if __name__ == "__main__":
    created = generate_all_samples()
    for p in created:
        print(f"Generated sample document: {p} ({p.stat().st_size} bytes)")
