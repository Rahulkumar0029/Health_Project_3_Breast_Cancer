import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation to add 'Page X of Y' and header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#2C3E50"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "CSE4001: Applications of ML in Industry — Health-Project 3: Breast Cancer Detection")
            self.setStrokeColor(colors.HexColor("#BDC3C7"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        self.drawString(54, 36, "Confidential — Academic Lab Experiment Report")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        
        self.restoreState()

def get_theme_styles():
    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#1A365D")   # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate Blue
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal body
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568")
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=dark_neutral,
        spaceAfter=5,
        alignment=TA_JUSTIFY
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3,
        alignment=TA_LEFT
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=dark_neutral
    )

    return {
        "styles": styles,
        "title": title_style,
        "subtitle": subtitle_style,
        "meta": meta_style,
        "h1": h1_style,
        "h2": h2_style,
        "body": body_style,
        "bullet": bullet_style,
        "code": code_style,
        "th": table_header_style,
        "td": table_cell_style,
        "primary": primary_color,
        "secondary": secondary_color,
        "border": border_color,
        "light_bg": colors.HexColor("#F7FAFC")
    }

def build_experiment_1_pdf():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base_dir, "experiment_pdfs")
    pdf_path = os.path.join(pdf_dir, "Experiment_1_Objective_Definition.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    t = get_theme_styles()

    story = []
    story.append(Paragraph("Health-Project 3: Breast Cancer Detection", t["subtitle"]))
    story.append(Paragraph("Experiment 1: Objective Definition Document", t["title"]))
    story.append(Paragraph("<b>Course:</b> CSE4001 — Applications of Machine Learning in Industry | <b>Target:</b> Malignant (1) vs. Benign (0)", t["meta"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=t["primary"], spaceAfter=10))

    story.append(Paragraph("1. Problem Statement & Clinical Context", t["h1"]))
    story.append(Paragraph(
        "A diagnostic pathology laboratory processes high volumes of Fine-Needle Aspirate (FNA) biopsy samples for suspected breast cancer. "
        "Manual microscopic examination of nuclear morphological features is labor-intensive and subject to inter-observer variability. "
        "The objective is to formulate an automated second-opinion machine learning triage system to analyze digitized morphological features of cell nuclei "
        "and automatically flag likely-malignant tumours for priority pathologist review.",
        t["body"]
    ))

    story.append(Paragraph("2. Machine Learning Formulation", t["h1"]))
    spec_data = [
        [Paragraph("<b>Dimension</b>", t["th"]), Paragraph("<b>Specification</b>", t["th"])],
        [Paragraph("ML Experiment Type", t["td"]), Paragraph("Supervised Binary Classification", t["td"])],
        [Paragraph("Input Features", t["td"]), Paragraph("30 continuous nuclear morphological measurements (mean, SE, worst of 10 geometric features)", t["td"])],
        [Paragraph("Target Variable", t["td"]), Paragraph("Diagnosis: <code>0 = Benign</code>, <code>1 = Malignant</code> (Binary Flag)", t["td"])],
        [Paragraph("Primary Metric", t["td"]), Paragraph("<b>Malignant Class Recall / Sensitivity (&ge; 98%)</b> to minimize missed malignancies", t["td"])],
        [Paragraph("Secondary Metrics", t["td"]), Paragraph("Precision (&ge; 90%), ROC-AUC (&ge; 0.98), F1-Score, False Negative Rate (FNR)", t["td"])],
        [Paragraph("Deployment Latency", t["td"]), Paragraph("&lt; 200 ms per sample via RESTful FastAPI scoring service", t["td"])]
    ]
    table = Table(spec_data, colWidths=[150, 354])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Clinical Success Metrics & Rationale", t["h1"]))
    story.append(Paragraph(
        "In clinical oncology screening, a <b>False Negative</b> (classifying a malignant tumor as benign) carries fatal risk, "
        "as vital cancer therapy is delayed. Conversely, a <b>False Positive</b> results in routine pathologist review. "
        "Therefore, the ML optimization objective strictly prioritizes Malignant-class Recall over raw overall accuracy.",
        t["body"]
    ))

    story.append(Paragraph("4. Stakeholder Analysis", t["h1"]))
    stakeholders = [
        "<b>Pathologists & Cytotechnologists:</b> Clinical end-users receiving prioritized alerts and confidence scores.",
        "<b>Patients:</b> Benefit from accelerated turnaround times and reduced diagnostic error.",
        "<b>Laboratory Directors:</b> Ensure clinical QA, HIPAA/GDPR regulatory compliance, and auditability.",
        "<b>ML Engineering Team:</b> Responsible for training, validating, deploying, and monitoring the predictive pipeline."
    ]
    for s in stakeholders:
        story.append(Paragraph(f"• {s}", t["bullet"]))

    story.append(Paragraph("5. Operational Constraints & Governance", t["h1"]))
    constraints = [
        "<b>Second-Opinion Decision Support:</b> The system acts as assistive decision support and does not replace certified pathologist review.",
        "<b>Class Imbalance Management:</b> The ~63% benign to 37% malignant ratio requires threshold tuning and loss weighting.",
        "<b>Zero Data Leakage:</b> Preprocessing statistics must strictly derive from training partitions."
    ]
    for c in constraints:
        story.append(Paragraph(f"• {c}", t["bullet"]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Experiment 1 PDF generated at: {pdf_path}")

def build_experiment_2_pdf():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base_dir, "experiment_pdfs")
    outputs_dir = os.path.join(base_dir, "outputs")
    pdf_path = os.path.join(pdf_dir, "Experiment_2_Dataset_Details.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    t = get_theme_styles()

    story = []
    story.append(Paragraph("Health-Project 3: Breast Cancer Detection", t["subtitle"]))
    story.append(Paragraph("Experiment 2: Dataset Details & Data Audit", t["title"]))
    story.append(Paragraph("<b>Course:</b> CSE4001 — Applications of Machine Learning in Industry | <b>Source:</b> UCI Machine Learning Repository", t["meta"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=t["primary"], spaceAfter=10))

    story.append(Paragraph("1. Dataset Source & Provenance", t["h1"]))
    story.append(Paragraph(
        "The Wisconsin Diagnostic Breast Cancer (WDBC) dataset was created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (1995) at the University of Wisconsin. "
        "It consists of 569 fine-needle aspirate (FNA) biopsy samples characterizing cell nuclei morphology digitized from microscope slides.",
        t["body"]
    ))

    story.append(Paragraph("2. Dataset Dimensions & Schema", t["h1"]))
    schema_data = [
        [Paragraph("<b>Attribute Group</b>", t["th"]), Paragraph("<b>Count</b>", t["th"]), Paragraph("<b>Details / Description</b>", t["th"])],
        [Paragraph("Total Records", t["td"]), Paragraph("569", t["td"]), Paragraph("Individual patient biopsy image extractions", t["td"])],
        [Paragraph("Metadata Field", t["td"]), Paragraph("1 (<code>id</code>)", t["td"]), Paragraph("Unique patient sample identifier (non-predictive)", t["td"])],
        [Paragraph("Target Variable", t["td"]), Paragraph("1 (<code>diagnosis</code>)", t["td"]), Paragraph("Ground truth: Benign (<code>B</code>) or Malignant (<code>M</code>)", t["td"])],
        [Paragraph("Feature Measurements", t["td"]), Paragraph("10 base metrics", t["td"]), Paragraph("Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension", t["td"])],
        [Paragraph("Total Predictive Features", t["td"]), Paragraph("30 continuous", t["td"]), Paragraph("Each of 10 metrics calculated as Mean, Standard Error (SE), and Worst (mean of 3 largest values)", t["td"])],
        [Paragraph("Missing / Null Values", t["td"]), Paragraph("0 (0.0%)", t["td"]), Paragraph("Complete dataset verified across all 569 rows", t["td"])]
    ]
    table = Table(schema_data, colWidths=[120, 84, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Target Distribution & Class Balance", t["h1"]))
    chart_path = os.path.join(outputs_dir, "experiment2_class_distribution.png")
    if os.path.exists(chart_path):
        story.append(Image(chart_path, width=6.2*inch, height=2.8*inch))
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Target Distribution:</b> Benign = 357 (62.74%), Malignant = 212 (37.26%). "
        "The moderate class imbalance confirms that accuracy alone is insufficient and requires precision-recall optimization.",
        t["body"]
    ))

    story.append(Paragraph("4. Known Limitations & Data Quality Findings", t["h1"]))
    findings = [
        "<b>Zero Missing Entries:</b> Verified 0 null values across all 32 raw attributes.",
        "<b>No Duplicate Biopsies:</b> Verified 569 unique records without replication.",
        "<b>Exclusion of ID Column:</b> The sample ID field carries no predictive biological signal and will be dropped in Experiment 3.",
        "<b>Cohort Scope:</b> Historical single-center dataset from 1995; multi-center external validation is advised before broad clinical deployment."
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", t["bullet"]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Experiment 2 PDF generated at: {pdf_path}")

def build_experiment_3_pdf():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base_dir, "experiment_pdfs")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    pdf_path = os.path.join(pdf_dir, "Experiment_3_Data_Preprocessing.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    t = get_theme_styles()

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("Health-Project 3: Breast Cancer Detection", t["subtitle"]))
    story.append(Paragraph("Experiment 3: Data Pre-processing", t["title"]))
    story.append(Paragraph("<b>Course:</b> CSE4001 — Applications of Machine Learning in Industry | <b>Target:</b> Malignant (1) vs. Benign (0)", t["meta"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=t["primary"], spaceAfter=10))

    # 1. Aim
    story.append(Paragraph("1. Aim", t["h1"]))
    story.append(Paragraph(
        "To preprocess the Breast Cancer Wisconsin (Diagnostic) dataset by handling missing values, "
        "removing duplicate records, dropping non-predictive identifiers, encoding the categorical diagnosis target, "
        "analyzing class distribution and imbalance weights, standardizing numerical features via <code>StandardScaler</code>, "
        "and preparing stratified train-validation-test partitions for machine-learning classification and clinical deployment.",
        t["body"]
    ))

    # 2. Objectives
    story.append(Paragraph("2. Objectives", t["h1"]))
    objectives = [
        "<b>Data Ingestion & Hygiene:</b> Load raw data and audit for null values and duplicate records.",
        "<b>Identifier Removal:</b> Drop database metadata (<code>id</code> column and unnamed index artifacts) to eliminate non-clinical predictive bias.",
        "<b>Binary Target Encoding:</b> Convert diagnosis labels (<code>M</code> &rarr; 1, <code>B</code> &rarr; 0) ensuring consistent oncology positive-class convention.",
        "<b>Class Imbalance Analysis:</b> Quantify majority benign vs. minority malignant distribution and calculate clinical loss weights.",
        "<b>Stratified Partitioning:</b> Split dataset into 80% Training (455), 10% Validation (57), and 10% Testing (57) sets while strictly preserving class ratios.",
        "<b>Feature Standardization:</b> Apply Z-score standardization fitted strictly on training data to prevent data leakage.",
        "<b>Pipeline Persistence:</b> Save the fitted scaler (<code>breast_cancer_scaler.pkl</code>) and train-ready datasets for downstream modeling."
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", t["bullet"]))

    # 3. Dataset Characteristics Table
    story.append(Paragraph("3. Dataset Summary & Characteristics", t["h1"]))
    data_summary = [
        [Paragraph("<b>Metric / Property</b>", t["th"]), Paragraph("<b>Value</b>", t["th"]), Paragraph("<b>Clinical Description / Specification</b>", t["th"])],
        [Paragraph("Dataset Source", t["td"]), Paragraph("UCI ML / Kaggle", t["td"]), Paragraph("Breast Cancer Wisconsin (Diagnostic) Dataset", t["td"])],
        [Paragraph("Total Biopsy Samples", t["td"]), Paragraph("569", t["td"]), Paragraph("Digitized Fine-Needle Aspirate (FNA) nuclear images", t["td"])],
        [Paragraph("Raw Attributes", t["td"]), Paragraph("32 columns", t["td"]), Paragraph("1 ID, 1 Diagnosis Target, 30 Diagnostic Features", t["td"])],
        [Paragraph("Predictive Features", t["td"]), Paragraph("30 numeric", t["td"]), Paragraph("10 morphological metrics &times; 3 statistics (mean, se, worst)", t["td"])],
        [Paragraph("Target Classes", t["td"]), Paragraph("Binary (0, 1)", t["td"]), Paragraph("Class 0: Benign (357, 62.74%) | Class 1: Malignant (212, 37.26%)", t["td"])],
        [Paragraph("Missing & Duplicate", t["td"]), Paragraph("0 missing / 0 dup", t["td"]), Paragraph("100% complete records verified across all 569 instances", t["td"])]
    ]
    t_summary = Table(data_summary, colWidths=[120, 90, 294])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 8))

    # 4. Preprocessing Methodology & Steps
    story.append(Paragraph("4. Preprocessing Methodology & Workflow", t["h1"]))
    steps_text = [
        "<b>Step 1 — Data Cleaning:</b> Removed <code>id</code> column and any trailing unnamed columns. The ID field holds zero predictive biological signal.",
        "<b>Step 2 — Target Encoding:</b> Mapped diagnosis labels <code>{'M': 1, 'B': 0}</code>. Malignant is coded as 1 because false negatives in cancer diagnosis carry catastrophic clinical risk, requiring sensitive recall optimization in Experiments 5 & 6.",
        "<b>Step 3 — Class Imbalance Handling:</b> The dataset contains 357 benign (62.7%) and 212 malignant (37.3%) cases (imbalance ratio 1.68:1). Computed positive class weight <code>pos_weight = 1.684</code> for loss-weighted models.",
        "<b>Step 4 — Stratified Split:</b> Employed two-stage stratified splitting with <code>random_state=42</code> to guarantee identical ~62.7% / 37.3% class proportions across subsets (Train: 455, Val: 57, Test: 57).",
        "<b>Step 5 — Standardization & Data Leakage Prevention:</b> Standardized features via <i>z = (x - &mu;) / &sigma;</i>. Crucially, the scaler is <b>fitted exclusively on X_train</b>. Validation and test sets are transformed using the stored parameters to prevent information contamination."
    ]
    for st in steps_text:
        story.append(Paragraph(f"• {st}", t["bullet"]))

    # Split Distribution Table
    story.append(Spacer(1, 6))
    split_data = [
        [Paragraph("<b>Partition</b>", t["th"]), Paragraph("<b>Sample Size</b>", t["th"]), Paragraph("<b>Percentage</b>", t["th"]), Paragraph("<b>Benign (0)</b>", t["th"]), Paragraph("<b>Malignant (1)</b>", t["th"]), Paragraph("<b>Malignant Ratio</b>", t["th"])],
        [Paragraph("Training Set", t["td"]), Paragraph("455", t["td"]), Paragraph("80.0%", t["td"]), Paragraph("285", t["td"]), Paragraph("170", t["td"]), Paragraph("37.36%", t["td"])],
        [Paragraph("Validation Set", t["td"]), Paragraph("57", t["td"]), Paragraph("10.0%", t["td"]), Paragraph("36", t["td"]), Paragraph("21", t["td"]), Paragraph("36.84%", t["td"])],
        [Paragraph("Test Set", t["td"]), Paragraph("57", t["td"]), Paragraph("10.0%", t["td"]), Paragraph("36", t["td"]), Paragraph("21", t["td"]), Paragraph("36.84%", t["td"])],
        [Paragraph("<b>Full Dataset</b>", t["td"]), Paragraph("<b>569</b>", t["td"]), Paragraph("<b>100.0%</b>", t["td"]), Paragraph("<b>357</b>", t["td"]), Paragraph("<b>212</b>", t["td"]), Paragraph("<b>37.26%</b>", t["td"])]
    ]
    t_split = Table(split_data, colWidths=[90, 75, 75, 80, 84, 100])
    t_split.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["secondary"]),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [t["light_bg"], colors.white]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_split)
    story.append(Spacer(1, 10))

    # 5. Visualizations Section
    story.append(Paragraph("5. Experimental Visualizations & Validation", t["h1"]))
    
    chart1_path = os.path.join(outputs_dir, "experiment3_class_distribution.png")
    chart2_path = os.path.join(outputs_dir, "experiment3_scaling_comparison.png")

    if os.path.exists(chart1_path):
        story.append(Paragraph("<b>Figure 1:</b> Target Class Distribution and Stratified Split Balance Verification", t["h2"]))
        story.append(Image(chart1_path, width=6.8*inch, height=2.6*inch))
        story.append(Spacer(1, 8))

    if os.path.exists(chart2_path):
        story.append(Paragraph("<b>Figure 2:</b> Distribution Densities Before vs. After Feature Standardization (StandardScaler)", t["h2"]))
        story.append(Image(chart2_path, width=6.8*inch, height=4.2*inch))
        story.append(Spacer(1, 8))

    # 6. Experimental Code & Execution Output
    story.append(Paragraph("6. Preprocessing Execution & Verification Code", t["h1"]))
    code_block = (
        "# Core Scaler Fitting (fitted ONLY on Training Data to prevent leakage)\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_val_scaled   = scaler.transform(X_val)\n"
        "X_test_scaled  = scaler.transform(X_test)\n\n"
        "# Preprocessing Pipeline Artifact Persistence\n"
        "import joblib\n"
        "joblib.dump(scaler, 'outputs/breast_cancer_scaler.pkl')\n"
        "# Saved Processed Partitions to data/processed/ (X_train, X_val, X_test, y_train, y_val, y_test)"
    )
    t_code = Table([[Paragraph(code_block.replace('\n', '<br/>'), t["code"])]], colWidths=[504])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_code)
    story.append(Spacer(1, 10))

    # 7. Observations & Key Insights
    story.append(Paragraph("7. Key Observations & Clinical Insights", t["h1"]))
    obs = [
        "<b>No Imputation Necessary for Baseline:</b> The 569 FNA biopsy records are 100% complete without missing values or duplicate instances.",
        "<b>Metadata Exclusion:</b> Removing patient ID numbers prevents spurious memorization and model distortion.",
        "<b>Preserved Stratification:</b> The stratified partition perfectly matched the 62.7% : 37.3% ratio across all sub-splits, avoiding partition bias.",
        "<b>Elimination of Scale Dominance:</b> Raw variables spanned disparate magnitudes (e.g., <code>area_mean</code> &le; 2500 vs. <code>smoothness_mean</code> &le; 0.16). Standardization brings all 30 features into &mu;=0, &sigma;=1, ensuring equal gradient contributions.",
        "<b>Zero Data Leakage:</b> Fitting the scaler strictly on X_train ensures that test set evaluation in Experiment 6 reflects true clinical out-of-sample performance.",
        "<b>Exported Pipeline Readiness:</b> The persisted scaler artifact (<code>breast_cancer_scaler.pkl</code>) guarantees identical numerical transformations during FastAPI production scoring (Experiment 7)."
    ]
    for o in obs:
        story.append(Paragraph(f"• {o}", t["bullet"]))

    # 8. Result & Conclusion
    story.append(Paragraph("8. Result", t["h1"]))
    story.append(Paragraph(
        "The Breast Cancer Wisconsin (Diagnostic) dataset was successfully cleaned, encoded, and preprocessed. "
        "The 30 diagnostic features were standardized using <code>StandardScaler</code> fitted strictly on the 455-sample training set, "
        "and stratified validation (57) and test (57) partitions were created. "
        "The trained scaler pipeline and processed CSV artifacts were successfully persisted in <code>outputs/</code> and <code>data/processed/</code>.",
        t["body"]
    ))

    story.append(Paragraph("9. Conclusion", t["h1"]))
    story.append(Paragraph(
        "Experiment 3 established an auditable, robust, and leakage-free data preprocessing pipeline for the breast cancer detection system. "
        "The dataset is now in clean, standardized format and ready for Exploratory Data Analysis (Experiment 4), "
        "Baseline Model Training with Logistic Regression, SVM, and Random Forest (Experiment 5), and subsequent clinical deployment.",
        t["body"]
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Experiment 3 PDF generated at: {pdf_path}")

def build_experiment_4_pdf():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base_dir, "experiment_pdfs")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    pdf_path = os.path.join(pdf_dir, "Experiment_4_Data_Exploration_EDA.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    t = get_theme_styles()

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("Health-Project 3: Breast Cancer Detection", t["subtitle"]))
    story.append(Paragraph("Experiment 4: Data Exploration (EDA)", t["title"]))
    story.append(Paragraph("<b>Course:</b> CSE4001 — Applications of Machine Learning in Industry | <b>Target:</b> Malignant (1) vs. Benign (0)", t["meta"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=t["primary"], spaceAfter=10))

    # 1. Aim
    story.append(Paragraph("1. Aim", t["h1"]))
    story.append(Paragraph(
        "To perform Exploratory Data Analysis (EDA) on the Breast Cancer Wisconsin (Diagnostic) "
        "dataset to understand feature distributions, identify relationships between features, detect "
        "outliers, and compare the characteristics of benign and malignant tumours.",
        t["body"]
    ))

    # 2. Objectives
    story.append(Paragraph("2. Objectives", t["h1"]))
    objectives = [
        "Analyze the distribution of the 30 numerical features across benign and malignant cohorts.",
        "Compare feature values between benign and malignant cases to detect morphometric discriminators.",
        "Study correlations between different features and identify multicollinear groups.",
        "Detect possible outliers using IQR analysis and boxplot visualizations.",
        "Visualize important features using publication-quality charts (histograms, boxplots, heatmaps, scatter plots).",
        "Compare features such as radius, texture, perimeter, and area between the two classes.",
        "Identify important patterns that can guide model selection in Experiment 5."
    ]
    for idx, obj in enumerate(objectives, 1):
        story.append(Paragraph(f"<b>{idx}.</b> {obj}", t["bullet"]))

    # 3. Dataset Characteristics Table
    story.append(Paragraph("3. Dataset Summary", t["h1"]))
    data_summary = [
        [Paragraph("<b>Metric / Property</b>", t["th"]), Paragraph("<b>Value</b>", t["th"]), Paragraph("<b>Description / Clinical Context</b>", t["th"])],
        [Paragraph("Dataset Name", t["td"]), Paragraph("WDBC (Wisconsin)", t["td"]), Paragraph("Digitized Fine-Needle Aspirate (FNA) nuclear images", t["td"])],
        [Paragraph("Sample Count (N)", t["td"]), Paragraph("569 biopsy samples", t["td"]), Paragraph("357 Benign (62.74%) | 212 Malignant (37.26%)", t["td"])],
        [Paragraph("Feature Space", t["td"]), Paragraph("30 continuous", t["td"]), Paragraph("10 nuclear geometries &times; 3 statistics (mean, se, worst)", t["td"])],
        [Paragraph("Target Variable", t["td"]), Paragraph("Binary (B/0, M/1)", t["td"]), Paragraph("0: Benign (negative) | 1: Malignant (positive)", t["td"])],
        [Paragraph("Class Imbalance", t["td"]), Paragraph("1.68 : 1 ratio", t["td"]), Paragraph("Requires recall & ROC-AUC optimization over raw accuracy", t["td"])]
    ]
    t_summary = Table(data_summary, colWidths=[120, 110, 274])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 8))

    # 4. Class-Wise Nuclear Morphometry Comparison Table
    story.append(Paragraph("4. Class-Wise Nuclear Morphometric Differences", t["h1"]))
    morph_data = [
        [Paragraph("<b>Nuclear Feature</b>", t["th"]), Paragraph("<b>Benign Mean (B)</b>", t["th"]), Paragraph("<b>Malignant Mean (M)</b>", t["th"]), Paragraph("<b>Relative Increase</b>", t["th"]), Paragraph("<b>Clinical Implication</b>", t["th"])],
        [Paragraph("<code>radius_mean</code>", t["td"]), Paragraph("12.15 &mu;m", t["td"]), Paragraph("17.46 &mu;m", t["td"]), Paragraph("+43.7%", t["td"]), Paragraph("Nuclear enlargement (anisonucleosis)", t["td"])],
        [Paragraph("<code>texture_mean</code>", t["td"]), Paragraph("17.92", t["td"]), Paragraph("21.61", t["td"]), Paragraph("+20.6%", t["td"]), Paragraph("Coarser chromatin clumping", t["td"])],
        [Paragraph("<code>perimeter_mean</code>", t["td"]), Paragraph("78.08 &mu;m", t["td"]), Paragraph("115.37 &mu;m", t["td"]), Paragraph("+47.8%", t["td"]), Paragraph("Irregular nuclear boundary stretch", t["td"])],
        [Paragraph("<code>area_mean</code>", t["td"]), Paragraph("462.79 &mu;m&sup2;", t["td"]), Paragraph("978.38 &mu;m&sup2;", t["td"]), Paragraph("<b>+111.4%</b>", t["td"]), Paragraph("Massive nuclear hypertrophy", t["td"])],
        [Paragraph("<code>concavity_mean</code>", t["td"]), Paragraph("0.046", t["td"]), Paragraph("0.161", t["td"]), Paragraph("<b>+250.0%</b>", t["td"]), Paragraph("Severe contour indentations & creases", t["td"])],
        [Paragraph("<code>concave_points_mean</code>", t["td"]), Paragraph("0.026", t["td"]), Paragraph("0.088", t["td"]), Paragraph("<b>+238.5%</b>", t["td"]), Paragraph("High frequency of structural folds", t["td"])],
        [Paragraph("<code>compactness_mean</code>", t["td"]), Paragraph("0.080", t["td"]), Paragraph("0.145", t["td"]), Paragraph("+81.3%", t["td"]), Paragraph("Deviations from spherical symmetry", t["td"])]
    ]
    t_morph = Table(morph_data, colWidths=[100, 75, 75, 74, 180])
    t_morph.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["secondary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_morph)
    story.append(Spacer(1, 8))

    # 5. Visualizations Section
    story.append(Paragraph("5. Visualizations & Analytical Charts", t["h1"]))

    chart_class = os.path.join(outputs_dir, "experiment4_class_distribution.png")
    chart_box = os.path.join(outputs_dir, "experiment4_boxplots_by_diagnosis.png")
    chart_heat = os.path.join(outputs_dir, "experiment4_correlation_heatmap.png")
    chart_scatter = os.path.join(outputs_dir, "experiment4_radius_vs_area_scatter.png")
    chart_target = os.path.join(outputs_dir, "experiment4_target_correlations.png")
    chart_means = os.path.join(outputs_dir, "experiment4_classwise_means.png")

    if os.path.exists(chart_box):
        story.append(Paragraph("<b>Figure 1:</b> Morphological Distribution Boxplots: Benign vs. Malignant Tumours", t["h2"]))
        story.append(Image(chart_box, width=6.8*inch, height=3.2*inch))
        story.append(Spacer(1, 6))

    if os.path.exists(chart_scatter) and os.path.exists(chart_means):
        story.append(Paragraph("<b>Figure 2:</b> Bivariate Decision Separability (Radius vs. Area) & Class-Wise Averages", t["h2"]))
        scatter_img = Image(chart_scatter, width=3.3*inch, height=2.4*inch)
        means_img = Image(chart_means, width=3.3*inch, height=2.4*inch)
        pair_table = Table([[scatter_img, means_img]], colWidths=[252, 252])
        pair_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(pair_table)
        story.append(Spacer(1, 6))

    if os.path.exists(chart_heat):
        story.append(Paragraph("<b>Figure 3:</b> Pearson Correlation Heatmap across 30 Continuous Nuclear Measurements", t["h2"]))
        story.append(Image(chart_heat, width=6.6*inch, height=3.6*inch))
        story.append(Spacer(1, 6))

    if os.path.exists(chart_target):
        story.append(Paragraph("<b>Figure 4:</b> Top Nuclear Morphological Correlates with Malignancy (Target = 1)", t["h2"]))
        story.append(Image(chart_target, width=6.6*inch, height=3.0*inch))
        story.append(Spacer(1, 6))

    # 6. Multicollinearity & Correlation Analysis Table
    story.append(Paragraph("6. Multicollinear Feature Redundancy (|r| > 0.90)", t["h1"]))
    corr_table_data = [
        [Paragraph("<b>Feature Pair (Multicollinear)</b>", t["th"]), Paragraph("<b>Pearson (r)</b>", t["th"]), Paragraph("<b>Redundancy Rationale & Handling Strategy</b>", t["th"])],
        [Paragraph("<code>perimeter_mean</code> &harr; <code>radius_mean</code>", t["td"]), Paragraph("0.9979", t["td"]), Paragraph("Direct geometric dependence (P &approx; 2&pi;r). Handled via L2 regularization.", t["td"])],
        [Paragraph("<code>area_mean</code> &harr; <code>radius_mean</code>", t["td"]), Paragraph("0.9874", t["td"]), Paragraph("Direct geometric dependence (A &approx; &pi;r&sup2;). Multicollinear in linear models.", t["td"])],
        [Paragraph("<code>perimeter_worst</code> &harr; <code>radius_worst</code>", t["td"]), Paragraph("0.9937", t["td"]), Paragraph("Collinear in extreme tumor boundary measurements.", t["td"])],
        [Paragraph("<code>area_worst</code> &harr; <code>radius_worst</code>", t["td"]), Paragraph("0.9840", t["td"]), Paragraph("Collinear in worst-case nuclear size metrics.", t["td"])],
        [Paragraph("<code>concave_points_mean</code> &harr; <code>concavity_mean</code>", t["td"]), Paragraph("0.9214", t["td"]), Paragraph("Strong correlation between number and severity of nuclear contour creases.", t["td"])]
    ]
    t_corr = Table(corr_table_data, colWidths=[160, 64, 280])
    t_corr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_corr)
    story.append(Spacer(1, 8))

    # 7. Code Snippet
    story.append(Paragraph("7. Python EDA Execution Implementation", t["h1"]))
    code_block = (
        "# Correlation Matrix & Multicollinearity Filtering\n"
        "correlation = df.select_dtypes(include=np.number).corr()\n"
        "high_corr = [(c1, c2, correlation.loc[c1, c2]) for c1 in correlation.columns\n"
        "             for c2 in correlation.columns if c1 != c2 and abs(correlation.loc[c1, c2]) > 0.90]\n\n"
        "# Class-Wise Comparison & Target Correlation\n"
        "class_means = df.groupby('diagnosis')[features].mean()\n"
        "target_corr = df.corr(numeric_only=True)['diagnosis'].sort_values(ascending=False)"
    )
    t_code = Table([[Paragraph(code_block.replace('\n', '<br/>'), t["code"])]], colWidths=[504])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_code)
    story.append(Spacer(1, 10))

    # 8. Key Observations (all 8 from syllabus)
    story.append(Paragraph("8. Key Observations & Analytical Findings", t["h1"]))
    obs = [
        "<b>Cohort Demographics:</b> The dataset contains 569 samples belonging to two diagnosis classes: 357 benign (62.74%) and 212 malignant (37.26%).",
        "<b>Class Imbalance Rationale:</b> The classes are moderately imbalanced (1.68:1 ratio). Downstream evaluation must focus on Malignant Class Recall and ROC-AUC.",
        "<b>Pronounced Morphometric Disparity:</b> Features such as mean radius, perimeter, and area show large differences across classes; malignant nuclei exhibit +111.4% higher mean area.",
        "<b>Multicollinearity in Geometry:</b> Size-related features exhibit extreme correlations (r > 0.98 for radius, perimeter, and area), necessitating regularized loss or ensemble estimators.",
        "<b>Contour Indentation as Chief Correlate:</b> Concave points (worst: r=0.7936, mean: r=0.7766) and concavity exhibit the highest predictive correlations with malignancy.",
        "<b>Biological Outliers:</b> Boxplots reveal extreme observations in high-grade carcinoma instances (area_se in 65 samples); these are preserved as clinically valid signals.",
        "<b>Clear Bivariate Separability:</b> Feature pairings (e.g. Mean Radius vs Mean Area) demonstrate clear separation boundaries between benign and malignant cases.",
        "<b>Model Selection Recommendations for Exp 5:</b> High dimensionality and multicollinearity indicate that Regularized Logistic Regression (L2), Support Vector Machines (RBF Kernel), and Random Forest are ideal baseline candidates."
    ]
    for o in obs:
        story.append(Paragraph(f"• {o}", t["bullet"]))

    # 9. Result & Conclusion
    story.append(Paragraph("9. Result", t["h1"]))
    story.append(Paragraph(
        "Exploratory Data Analysis was successfully performed on the Breast Cancer Wisconsin dataset. "
        "The distributions of important tumour-related features were analyzed, relationships between numerical features "
        "were examined using correlation analysis, and differences between benign and malignant cases were visualized "
        "using boxplots, scatter plots, and heatmaps.",
        t["body"]
    ))

    story.append(Paragraph("10. Conclusion", t["h1"]))
    story.append(Paragraph(
        "The EDA reveals that several numerical measurements contain useful information for distinguishing between "
        "benign and malignant tumour samples. Strong correlations are also present among some features, indicating "
        "possible redundancy. These findings directly guide feature selection, regularization, and model training in Experiment 5.",
        t["body"]
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Experiment 4 PDF generated at: {pdf_path}")

def build_experiment_5_pdf():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_dir = os.path.join(base_dir, "experiment_pdfs")
    outputs_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "Experiment_5_Model_Training.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    t = get_theme_styles()
    story = []

    # Title & Metadata
    story.append(Paragraph("Experiment 5: Model Training, Hyperparameter Tuning & Cross-Validation", t["title"]))
    story.append(Paragraph("CSE4001: Applications of Machine Learning in Industry — Health-Project 3 (Breast Cancer)", t["subtitle"]))
    story.append(Spacer(1, 4))

    meta_data = [
        [Paragraph("<b>Course:</b> CSE4001 Applications of ML", t["meta"]), Paragraph("<b>Dataset:</b> WDBC (569 FNA Samples, 30 Features)", t["meta"])],
        [Paragraph("<b>Focus:</b> Model Training & Tuning", t["meta"]), Paragraph("<b>Partitions:</b> Train (80%, N=455), Val (10%, N=57), Test (10%, N=57)", t["meta"])],
        [Paragraph("<b>Algorithms:</b> Logistic Reg., SVM (RBF), Random Forest", t["meta"]), Paragraph("<b>Status:</b> Completed & Verified", t["meta"])]
    ]
    t_meta = Table(meta_data, colWidths=[252, 252])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), t["light_bg"]),
        ('BOX', (0,0), (-1,-1), 0.5, t["border"]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # 1. Aim
    story.append(Paragraph("1. Aim", t["h1"]))
    story.append(Paragraph(
        "To implement, train, hyperparameter-tune, and validate candidate machine learning classification algorithms "
        "(<b>Logistic Regression</b>, <b>Support Vector Machine</b>, and <b>Random Forest</b>) on the standardized "
        "Breast Cancer Wisconsin dataset using Stratified 5-Fold Cross-Validation, compare model performance against baseline heuristics, "
        "and serialize the optimal model checkpoints for downstream clinical evaluation.",
        t["body"]
    ))

    # 2. Objectives
    story.append(Paragraph("2. Objectives", t["h1"]))
    objs = [
        "<b>Establish Baseline Heuristics:</b> Implement a Zero-Rule Dummy classifier (majority class heuristic) and an untuned Logistic Regression baseline.",
        "<b>Train Candidate Classifiers:</b> Implement Regularized Logistic Regression (L2), Support Vector Classifier (RBF kernel), and Random Forest (ensemble bagging).",
        "<b>Hyperparameter Optimization:</b> Execute Stratified GridSearchCV (k=5) across regularizers (C), kernel coefficients (&gamma;), and tree depths.",
        "<b>Evaluate Multi-Metric Generalization:</b> Track balanced accuracy, malignant recall (sensitivity), specificity, precision, F1-score, and ROC-AUC.",
        "<b>Error & Confusion Matrix Analysis:</b> Evaluate models on the independent validation holdout set (N=57) to minimize False Negatives (missed cancer).",
        "<b>Feature Attribution & Interpretability:</b> Compare Random Forest Mean Decrease in Impurity (Gini) with Logistic Regression standardized weights.",
        "<b>Model Checkpoint Serialization:</b> Save trained model artifacts (.pkl) and training metadata JSON in the models directory."
    ]
    for obj in objs:
        story.append(Paragraph(f"• {obj}", t["bullet"]))
    story.append(Spacer(1, 6))

    # 3. Algorithm & Mathematical Background
    story.append(Paragraph("3. Mathematical Formulations of Candidate Algorithms", t["h1"]))
    math_data = [
        [Paragraph("<b>Model Architecture</b>", t["th"]), Paragraph("<b>Mathematical Formulation & Objective Function</b>", t["th"]), Paragraph("<b>Key Hyperparameters</b>", t["th"])],
        [
            Paragraph("<b>Logistic Regression (L2)</b>", t["td"]),
            Paragraph("P(y=1|x) = &sigma;(w<sup>T</sup>x + b) = 1 / (1 + e<sup>-(w<sup>T</sup>x + b)</sup>)<br/>"
                      "Loss = -1/N &sum; [y ln &sigma; + (1-y) ln(1-&sigma;)] + (1/2C) ||w||<sub>2</sub><sup>2</sup>", t["td"]),
            Paragraph("C = 1.0, penalty = 'l2', solver = 'lbfgs'", t["td"])
        ],
        [
            Paragraph("<b>SVM (RBF Kernel)</b>", t["td"]),
            Paragraph("min 1/2 ||w||<sup>2</sup> + C &sum; &xi;<sub>i</sub> s.t. y<sub>i</sub>(w<sup>T</sup>&phi;(x<sub>i</sub>)+b) &ge; 1 - &xi;<sub>i</sub><br/>"
                      "Kernel: K(x<sub>i</sub>, x<sub>j</sub>) = exp(-&gamma; ||x<sub>i</sub> - x<sub>j</sub>||<sup>2</sup>)", t["td"]),
            Paragraph("C = 5.0, &gamma; = 0.01, kernel = 'rbf'", t["td"])
        ],
        [
            Paragraph("<b>Random Forest</b>", t["td"]),
            Paragraph("Ensemble aggregation of B=50 decorrelated decision trees:<br/>"
                      "P&#770;(y=1|x) = (1/B) &sum; T<sub>b</sub>(x); Split via Gini I<sub>G</sub> = 1 - &sum; p<sub>k</sub><sup>2</sup>", t["td"]),
            Paragraph("n_est = 50, max_depth = 12, max_features = 'log2'", t["td"])
        ]
    ]
    t_math = Table(math_data, colWidths=[110, 264, 130])
    t_math.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_math)
    story.append(Spacer(1, 8))

    # 4. Cross-Validation Results Table
    story.append(Paragraph("4. 5-Fold Stratified Cross-Validation Benchmark", t["h1"]))
    cv_table_data = [
        [Paragraph("<b>Model Architecture</b>", t["th"]), Paragraph("<b>CV Accuracy</b>", t["th"]), Paragraph("<b>CV Recall (Malignant)</b>", t["th"]), Paragraph("<b>CV Precision</b>", t["th"]), Paragraph("<b>CV F1-Score</b>", t["th"]), Paragraph("<b>CV ROC-AUC</b>", t["th"])],
        [Paragraph("<b>Baseline (Majority)</b>", t["td"]), Paragraph("0.6264 &plusmn; 0.002", t["td"]), Paragraph("0.0000 &plusmn; 0.000", t["td"]), Paragraph("0.0000 &plusmn; 0.000", t["td"]), Paragraph("0.0000 &plusmn; 0.000", t["td"]), Paragraph("0.5000 &plusmn; 0.000", t["td"])],
        [Paragraph("<b>Untuned Logistic Reg.</b>", t["td"]), Paragraph("0.9714 &plusmn; 0.015", t["td"]), Paragraph("0.9471 &plusmn; 0.043", t["td"]), Paragraph("0.9758 &plusmn; 0.024", t["td"]), Paragraph("0.9608 &plusmn; 0.021", t["td"]), Paragraph("0.9942 &plusmn; 0.005", t["td"])],
        [Paragraph("<b>Logistic Regression (L2)</b>", t["td"]), Paragraph("<b>0.9736 &plusmn; 0.015</b>", t["td"]), Paragraph("<b>0.9529 &plusmn; 0.040</b>", t["td"]), Paragraph("0.9771 &plusmn; 0.028", t["td"]), Paragraph("0.9640 &plusmn; 0.021", t["td"]), Paragraph("<b>0.9958 &plusmn; 0.005</b>", t["td"])],
        [Paragraph("<b>SVM (RBF Kernel)</b>", t["td"]), Paragraph("<b>0.9758 &plusmn; 0.013</b>", t["td"]), Paragraph("0.9471 &plusmn; 0.043", t["td"]), Paragraph("<b>0.9889 &plusmn; 0.022</b>", t["td"]), Paragraph("<b>0.9666 &plusmn; 0.019</b>", t["td"]), Paragraph("<b>0.9960 &plusmn; 0.005</b>", t["td"])],
        [Paragraph("<b>Random Forest</b>", t["td"]), Paragraph("0.9582 &plusmn; 0.016", t["td"]), Paragraph("0.9412 &plusmn; 0.026", t["td"]), Paragraph("0.9469 &plusmn; 0.022", t["td"]), Paragraph("0.9439 &plusmn; 0.022", t["td"]), Paragraph("0.9913 &plusmn; 0.005", t["td"])],
    ]
    t_cv = Table(cv_table_data, colWidths=[124, 76, 80, 76, 74, 74])
    t_cv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_cv)
    story.append(Spacer(1, 8))

    # 5. Visualizations Section
    story.append(Paragraph("5. Visualizations & Performance Diagnostics", t["h1"]))

    chart_cv = os.path.join(outputs_dir, "experiment5_cv_comparison.png")
    chart_cm = os.path.join(outputs_dir, "experiment5_confusion_matrices.png")
    chart_roc = os.path.join(outputs_dir, "experiment5_roc_curves.png")
    chart_pr = os.path.join(outputs_dir, "experiment5_precision_recall_curves.png")
    chart_feat = os.path.join(outputs_dir, "experiment5_feature_importance.png")
    chart_learn = os.path.join(outputs_dir, "experiment5_learning_curves.png")
    chart_heat = os.path.join(outputs_dir, "experiment5_hyperparameter_heatmaps.png")

    if os.path.exists(chart_cv):
        story.append(Paragraph("<b>Figure 1:</b> 5-Fold Stratified Cross-Validation Benchmark Across Candidate Models", t["h2"]))
        story.append(Image(chart_cv, width=6.8*inch, height=3.3*inch))
        story.append(Spacer(1, 6))

    if os.path.exists(chart_cm):
        story.append(Paragraph("<b>Figure 2:</b> Confusion Matrices on Validation Holdout Set (N=57, Benign=36, Malignant=21)", t["h2"]))
        story.append(Image(chart_cm, width=6.8*inch, height=2.2*inch))
        story.append(Spacer(1, 6))

    if os.path.exists(chart_roc) and os.path.exists(chart_pr):
        story.append(Paragraph("<b>Figure 3:</b> Discrimination Diagnostics: Validation ROC Curves vs. Precision-Recall Curves", t["h2"]))
        roc_img = Image(chart_roc, width=3.3*inch, height=2.5*inch)
        pr_img = Image(chart_pr, width=3.3*inch, height=2.5*inch)
        pair_table = Table([[roc_img, pr_img]], colWidths=[252, 252])
        pair_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(pair_table)
        story.append(Spacer(1, 6))

    if os.path.exists(chart_feat):
        story.append(Paragraph("<b>Figure 4:</b> Model Interpretability: Random Forest Gini Importance vs. Logistic Regression Weights", t["h2"]))
        story.append(Image(chart_feat, width=6.8*inch, height=3.1*inch))
        story.append(Spacer(1, 6))

    if os.path.exists(chart_learn) and os.path.exists(chart_heat):
        story.append(Paragraph("<b>Figure 5:</b> Training Generalization Stability (Learning Curves) & SVM Hyperparameter Surface", t["h2"]))
        learn_img = Image(chart_learn, width=3.6*inch, height=2.2*inch)
        heat_img = Image(chart_heat, width=3.0*inch, height=2.2*inch)
        pair_table2 = Table([[learn_img, heat_img]], colWidths=[270, 234])
        pair_table2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(pair_table2)
        story.append(Spacer(1, 6))

    # 6. Validation Holdout Set Results Table
    story.append(Paragraph("6. Validation Set (N=57) Performance & Clinical Error Trade-Offs", t["h1"]))
    val_table_data = [
        [Paragraph("<b>Model Architecture</b>", t["th"]), Paragraph("<b>Val Accuracy</b>", t["th"]), Paragraph("<b>Recall (Sensitivity)</b>", t["th"]), Paragraph("<b>Specificity</b>", t["th"]), Paragraph("<b>Precision</b>", t["th"]), Paragraph("<b>F1-Score</b>", t["th"]), Paragraph("<b>ROC-AUC</b>", t["th"])],
        [Paragraph("<b>Logistic Regression</b>", t["td"]), Paragraph("94.74%", t["td"]), Paragraph("85.71% (FN=3)", t["td"]), Paragraph("<b>100.00% (FP=0)</b>", t["td"]), Paragraph("100.00%", t["td"]), Paragraph("0.9231", t["td"]), Paragraph("0.9934", t["td"])],
        [Paragraph("<b>SVM (RBF Kernel)</b>", t["td"]), Paragraph("<b>96.49%</b>", t["td"]), Paragraph("90.48% (FN=2)", t["td"]), Paragraph("<b>100.00% (FP=0)</b>", t["td"]), Paragraph("100.00%", t["td"]), Paragraph("0.9500", t["td"]), Paragraph("0.9934", t["td"])],
        [Paragraph("<b>Random Forest</b>", t["td"]), Paragraph("<b>96.49%</b>", t["td"]), Paragraph("<b>95.24% (FN=1)</b>", t["td"]), Paragraph("97.22% (FP=1)", t["td"]), Paragraph("95.24%", t["td"]), Paragraph("<b>0.9524</b>", t["td"]), Paragraph("<b>0.9947</b>", t["td"])],
    ]
    t_val = Table(val_table_data, colWidths=[124, 66, 84, 80, 56, 48, 46])
    t_val.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), t["primary"]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, t["border"]),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [t["light_bg"], colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_val)
    story.append(Spacer(1, 8))

    # 7. Code Snippet
    story.append(Paragraph("7. Model Training & GridSearchCV Implementation Snippet", t["h1"]))
    code_block = (
        "# Stratified 5-Fold Grid Search Optimization\n"
        "cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
        "grid = GridSearchCV(estimator=SVC(probability=True, random_state=42),\n"
        "                    param_grid={'C': [0.1, 1.0, 5.0, 10.0], 'gamma': ['scale', 'auto', 0.01],\n"
        "                                'kernel': ['rbf', 'linear'], 'class_weight': ['balanced', None]},\n"
        "                    cv=cv_strategy, scoring=['accuracy', 'recall', 'f1', 'roc_auc'], refit='roc_auc')\n"
        "grid.fit(X_train_scaled, y_train)\n"
        "joblib.dump(grid.best_estimator_, 'models/svm_rbf_best.pkl')"
    )
    t_code = Table([[Paragraph(code_block.replace('\n', '<br/>'), t["code"])]], colWidths=[504])
    t_code.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_code)
    story.append(Spacer(1, 10))

    # 8. Key Observations
    story.append(Paragraph("8. Key Observations & Clinical Findings", t["h1"]))
    obs = [
        "<b>Decisive Baseline Outperformance:</b> All candidate models scored &gt;0.991 ROC-AUC, demonstrating immense discriminative power compared to naive baseline (0.500 AUC).",
        "<b>SVM RBF Peak CV Score:</b> Support Vector Machine achieved the highest CV score (0.9960 ROC-AUC, 97.58% Accuracy, 98.89% Precision), effectively resolving non-linear nuclear boundaries.",
        "<b>Random Forest Clinical Sensitivity:</b> Random Forest achieved the highest Malignant Recall on the validation holdout set (95.24%, FN=1), minimizing the risk of missed cancer diagnosis.",
        "<b>Validation Specificity:</b> SVM and Logistic Regression achieved 100.00% Specificity (FP=0), completely eliminating false-positive biopsy referrals on the validation set.",
        "<b>Feature Alignment with EDA:</b> Top predictive features (concave_points_worst, radius_worst, perimeter_worst) aligned perfectly with Experiment 4 Pearson correlation rankings.",
        "<b>Checkpoint Preservation:</b> Serialized model pipelines (logistic_regression_best.pkl, svm_rbf_best.pkl, random_forest_best.pkl) were saved to models/ directory."
    ]
    for o in obs:
        story.append(Paragraph(f"• {o}", t["bullet"]))

    # 9. Result & Conclusion
    story.append(Paragraph("9. Result", t["h1"]))
    story.append(Paragraph(
        "Candidate classification models (Logistic Regression, SVM, and Random Forest) were successfully trained, "
        "hyperparameter-tuned via Stratified 5-Fold Cross-Validation, and validated on the standardized Breast Cancer Wisconsin dataset. "
        "Cross-validation benchmarks, validation holdout metrics, diagnostic plots (ROC, PR, learning curves, feature importances), "
        "and serialized model artifacts were generated and saved.",
        t["body"]
    ))

    story.append(Paragraph("10. Conclusion", t["h1"]))
    story.append(Paragraph(
        "Hyperparameter tuning established SVM with RBF kernel and Random Forest as superior classifiers, achieving over 96.49% "
        "validation accuracy and &gt;0.993 ROC-AUC. Random Forest provided the highest clinical sensitivity (95.24% recall), "
        "while SVM demonstrated flawless specificity (100%). These saved model artifacts form the foundation for ensemble modeling, "
        "decision threshold calibration, and comprehensive error analysis in Experiment 6.",
        t["body"]
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Experiment 5 PDF generated at: {pdf_path}")

if __name__ == "__main__":
    build_experiment_1_pdf()
    build_experiment_2_pdf()
    build_experiment_3_pdf()
    build_experiment_4_pdf()
    build_experiment_5_pdf()


