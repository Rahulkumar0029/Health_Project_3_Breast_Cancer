"""
Generates the official Word Document (.docx) for Experiment 5: Model Training
Formatted strictly according to the CSE4001 Lab Manual and academic submission standards.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
            f'  <w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def create_experiment_5_docx():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(base_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    doc_path = os.path.join(report_dir, "Experiment_5_Model_Training.docx")

    doc = Document()

    # Page Margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

        # Header
        header = section.header
        header_p = header.paragraphs[0]
        header_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        header_run = header_p.add_run("Rahul Kumar - 230160223017")
        header_run.font.name = 'Arial'
        header_run.font.size = Pt(9.5)
        header_run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

        # Footer
        footer = section.footer
        footer_p = footer.paragraphs[0]
        footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer_run = footer_p.add_run("Rahul Kumar - 230160223017")
        footer_run.font.name = 'Arial'
        footer_run.font.size = Pt(9.5)
        footer_run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # Document Title
    p_title1 = doc.add_paragraph()
    p_title1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title1.paragraph_format.space_before = Pt(0)
    p_title1.paragraph_format.space_after = Pt(2)
    run_exp = p_title1.add_run("Experiment 5")
    run_exp.bold = True
    run_exp.font.size = Pt(15)
    run_exp.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    p_title2 = doc.add_paragraph()
    p_title2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title2.paragraph_format.space_before = Pt(0)
    p_title2.paragraph_format.space_after = Pt(14)
    run_sub = p_title2.add_run("Model Training and Hyperparameter Tuning")
    run_sub.bold = True
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    def add_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(11)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.font.size = Pt(9.5)
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(2.5)
        r_bold = p.add_run(bold_prefix + " ")
        r_bold.bold = True
        r_bold.font.size = Pt(9.5)
        r_text = p.add_run(text)
        r_text.font.size = Pt(9.5)
        return p

    def format_table(table, col_widths, headers, rows_data):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        # Format Header
        hdr_cells = table.rows[0].cells
        for i, title in enumerate(headers):
            hdr_cells[i].text = title
            p = hdr_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
            set_cell_background(hdr_cells[i], "EFEFEF")
            set_cell_margins(hdr_cells[i], top=80, bottom=80, left=100, right=100)

        # Format Data Rows
        for r_idx, row_data in enumerate(rows_data):
            row_cells = table.rows[r_idx + 1].cells
            for c_idx, val in enumerate(row_data):
                row_cells[c_idx].text = str(val)
                p = row_cells[c_idx].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(8.5)
                    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                set_cell_margins(row_cells[c_idx], top=60, bottom=60, left=100, right=100)
                if r_idx % 2 == 1:
                    set_cell_background(row_cells[c_idx], "F9F9F9")

        # Set Column Widths
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)

        set_table_borders(table, color="CCCCCC", sz="4", val="single")

    # ---------------------------------------------------------
    # 1. Objective
    # ---------------------------------------------------------
    add_heading("1. Objective")
    add_body(
        "This experiment implements, trains, hyperparameter-tunes, and evaluates candidate supervised machine learning "
        "algorithms (Logistic Regression, Support Vector Machine, and Random Forest) on the standardized Breast Cancer Wisconsin "
        "(Diagnostic) dataset. It utilizes Stratified 5-Fold Cross-Validation (GridSearchCV) to optimize classification metrics "
        "with an emphasis on Malignant Recall (sensitivity) and ROC-AUC, compares performance against baseline heuristics, and "
        "saves the serialized model checkpoints to establish reproducible artifacts for downstream evaluation."
    )

    # ---------------------------------------------------------
    # 2. Baseline Model Formulation
    # ---------------------------------------------------------
    add_heading("2. Baseline Model Formulation and Lower-Bound Benchmarks")
    add_body(
        "Zero-rule heuristic baseline: A DummyClassifier utilizing the 'most_frequent' strategy was trained on the training partition "
        "(455 instances). Because benign cases form 62.6% of the training cohort, the majority-class baseline achieves an accuracy of "
        "63.16% on the validation set, but yields a Malignant Recall of 0.00% and a default ROC-AUC of 0.5000. This confirms that naive "
        "accuracy is clinically unusable and establishes the true empirical lower bound."
    )
    add_body(
        "Untuned standard linear baseline: A default Logistic Regression model without hyperparameter optimization was evaluated, "
        "yielding 94.74% validation accuracy, 85.71% malignant recall, and 0.9934 ROC-AUC. Candidate models are required to exceed these "
        "thresholds under cross-validation."
    )

    # ---------------------------------------------------------
    # 3. Candidate Algorithms & Architecture Specification
    # ---------------------------------------------------------
    add_heading("3. Candidate Algorithms and Architectural Specifications")
    add_body(
        "Three distinct algorithm families were selected to balance parametric interpretability, non-linear geometric margin separation, "
        "and tree-based ensemble variance reduction:"
    )

    add_bullet(
        "Logistic Regression (L2 Regularized):",
        "Linear parametric classifier using the logistic sigmoid activation P(y=1|x) = 1 / (1 + e^-(w^T x + b)) with an L2 Ridge penalty "
        "(1/2C * ||w||_2^2) integrated into binary cross-entropy loss to control multicollinearity."
    )
    add_bullet(
        "Support Vector Machine (SVM - RBF Kernel):",
        "Maximum-margin classifier finding an optimal separating hyperplane in a high-dimensional reproducing Hilbert space using "
        "the Radial Basis Function kernel K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)."
    )
    add_bullet(
        "Random Forest Classifier:",
        "Non-parametric bootstrap aggregation (Bagging) ensemble of decorrelated decision trees with random feature subspace sampling "
        "(sqrt/log2) and Gini impurity node splitting criteria."
    )

    # ---------------------------------------------------------
    # 4. Hyperparameter Optimization Setup (GridSearchCV)
    # ---------------------------------------------------------
    add_heading("4. Hyperparameter Optimization Setup (Stratified GridSearchCV)")
    add_body(
        "Hyperparameters were tuned via 5-Fold Stratified Cross-Validation on the 455 training instances. The optimization target was "
        "set to ROC-AUC ('refit=roc_auc') to maximize global discrimination capability across potential diagnostic decision thresholds."
    )

    t_grid = doc.add_table(rows=4, cols=4)
    grid_headers = ["Algorithm", "Search Parameter Grid", "Optimal Hyperparameters", "Tuning Duration"]
    grid_rows = [
        ["Logistic Regression", "C: [0.01 to 10.0], solver: ['lbfgs', 'liblinear'], class_weight: ['balanced', None]", "C: 1.0, penalty: 'l2', solver: 'lbfgs', class_weight: None", "7.74 s"],
        ["Support Vector Machine", "C: [0.1 to 50.0], gamma: ['scale', 'auto', 0.01, 0.05], kernel: ['rbf', 'linear']", "C: 5.0, gamma: 0.01, kernel: 'rbf', class_weight: None", "6.18 s"],
        ["Random Forest", "n_estimators: [50, 100, 200], max_depth: [3 to 12, None], max_features: ['sqrt', 'log2']", "n_estimators: 50, max_depth: 12, max_features: 'log2', balanced", "221.04 s"],
    ]
    format_table(t_grid, [1.4, 2.1, 2.0, 0.9], grid_headers, grid_rows)

    # ---------------------------------------------------------
    # 5. Stratified 5-Fold Cross-Validation Performance
    # ---------------------------------------------------------
    add_heading("5. Stratified 5-Fold Cross-Validation Performance Benchmark")
    add_body(
        "The cross-validation scores (Mean ± Standard Deviation) across 5 stratified folds on the training set (N=455) are summarized below:"
    )

    t_cv = doc.add_table(rows=6, cols=6)
    cv_headers = ["Model Architecture", "CV Accuracy", "CV Recall (Malignant)", "CV Precision", "CV F1-Score", "CV ROC-AUC"]
    cv_rows = [
        ["Baseline (Majority)", "0.6264 ± 0.002", "0.0000 ± 0.000", "0.0000 ± 0.000", "0.0000 ± 0.000", "0.5000 ± 0.000"],
        ["Untuned Logistic Reg.", "0.9714 ± 0.015", "0.9471 ± 0.043", "0.9758 ± 0.024", "0.9608 ± 0.021", "0.9942 ± 0.005"],
        ["Logistic Regression (L2)", "0.9736 ± 0.015", "0.9529 ± 0.040", "0.9771 ± 0.028", "0.9640 ± 0.021", "0.9958 ± 0.005"],
        ["SVM (RBF Kernel)", "0.9758 ± 0.013", "0.9471 ± 0.043", "0.9889 ± 0.022", "0.9666 ± 0.019", "0.9960 ± 0.005"],
        ["Random Forest", "0.9582 ± 0.016", "0.9412 ± 0.026", "0.9469 ± 0.022", "0.9439 ± 0.022", "0.9913 ± 0.005"],
    ]
    format_table(t_cv, [1.5, 0.95, 1.1, 0.95, 0.95, 0.95], cv_headers, cv_rows)

    # ---------------------------------------------------------
    # 6. Validation Holdout Set Evaluation & Error Analysis
    # ---------------------------------------------------------
    add_heading("6. Validation Holdout Set Evaluation and Clinical Error Trade-Offs")
    add_body(
        "Performance of the optimal model checkpoints on the independent validation holdout set (N=57: 36 Benign, 21 Malignant):"
    )

    t_val = doc.add_table(rows=4, cols=8)
    val_headers = ["Model", "Accuracy", "Recall", "Specificity", "Precision", "F1", "ROC-AUC", "Confusion Matrix"]
    val_rows = [
        ["Logistic Regression", "94.74%", "85.71%", "100.00%", "100.00%", "0.9231", "0.9934", "TP=18, FN=3, TN=36, FP=0"],
        ["SVM (RBF Kernel)", "96.49%", "90.48%", "100.00%", "100.00%", "0.9500", "0.9934", "TP=19, FN=2, TN=36, FP=0"],
        ["Random Forest", "96.49%", "95.24%", "97.22%", "95.24%", "0.9524", "0.9947", "TP=20, FN=1, TN=35, FP=1"],
    ]
    format_table(t_val, [1.3, 0.7, 0.7, 0.75, 0.7, 0.65, 0.75, 1.35], val_headers, val_rows)

    # ---------------------------------------------------------
    # 7. Checkpoint Serialization & Artifacts
    # ---------------------------------------------------------
    add_heading("7. Model Checkpoint Serialization and Saved Artifacts")
    add_body(
        "Trained models and evaluation metadata were serialized to the 'models/' directory for deployment and Experiment 6 evaluation:"
    )

    t_art = doc.add_table(rows=5, cols=4)
    art_headers = ["Artifact File Name", "Saved Object / Algorithm", "Format & Size", "Clinical / Evaluation Utility"]
    art_rows = [
        ["logistic_regression_best.pkl", "L2 Logistic Regression (C=1.0)", "joblib (.pkl) | 1.81 KB", "Fast linear inference and interpretable odds ratios."],
        ["svm_rbf_best.pkl", "Support Vector Classifier (RBF)", "joblib (.pkl) | 18.37 KB", "Optimal decision boundary with 100% specificity."],
        ["random_forest_best.pkl", "Random Forest (50 Trees, Balanced)", "joblib (.pkl) | 165.45 KB", "Top sensitivity (95.24% recall, FN=1) for screening triage."],
        ["model_training_summary.json", "JSON Experiment Metrics Manifest", "JSON Text | 2.67 KB", "Machine-readable record of hyperparameters and CV scores."],
    ]
    format_table(t_art, [1.6, 1.7, 1.4, 1.7], art_headers, art_rows)

    # ---------------------------------------------------------
    # 8. Feature Attribution & Model Interpretability
    # ---------------------------------------------------------
    add_heading("8. Feature Attribution and Model Interpretability")
    add_body(
        "Model interpretability analysis confirms strong alignment with Experiment 4 Exploratory Data Analysis findings:"
    )
    add_bullet(
        "Random Forest Gini Importances:",
        "The top predictive features driving tree splits were concave_points_worst (0.162), perimeter_worst (0.134), "
        "radius_worst (0.118), area_worst (0.098), and concavity_mean (0.076)."
    )
    add_bullet(
        "Logistic Regression Standardized Weights:",
        "The largest positive coefficients (increasing cancer odds) were concave_points_worst (+1.18), radius_worst (+1.04), "
        "and texture_worst (+0.92), verifying that irregular tumor contour creases and severe size increase are primary cancer signals."
    )

    # ---------------------------------------------------------
    # 9. Known Modeling Limitations & Error Analysis
    # ---------------------------------------------------------
    add_heading("9. Known Modeling Limitations and Risk Mitigation")
    add_bullet(
        "False Negative Severity:",
        "In oncology triage, missed cancer cases (False Negatives) carry severe prognosis penalties. Random Forest minimized FN to 1, "
        "while Logistic Regression produced 3 FNs under the default 0.50 threshold, requiring decision threshold calibration in Experiment 6."
    )
    add_bullet(
        "Computational Complexity:",
        "Random Forest required 221.04 seconds for grid search over 270 fit candidates, whereas SVM and Logistic Regression completed in <8 seconds."
    )
    add_bullet(
        "Kernel Calibration:",
        "SVM with RBF kernel relies on Platt scaling for posterior probability generation; confidence calibration must be audited before clinical deployment."
    )

    # ---------------------------------------------------------
    # 10. Conclusion
    # ---------------------------------------------------------
    add_heading("10. Conclusion")
    add_body(
        "Experiment 5 successfully trained, hyperparameter-tuned, and cross-validated three candidate classifiers on the Breast Cancer "
        "Wisconsin dataset. SVM with RBF kernel achieved the highest cross-validation benchmark (0.9960 ROC-AUC, 97.58% accuracy, 98.89% precision), "
        "while Random Forest achieved the highest clinical sensitivity (95.24% malignant recall, 1 False Negative) on the validation holdout set. "
        "All model checkpoints and metadata manifests have been serialized and validated, establishing a robust foundation for ensemble modeling, "
        "threshold calibration, and error analysis in Experiment 6."
    )

    # Save document
    doc.save(doc_path)
    print(f"[SUCCESS] Experiment 5 Word Document (.docx) generated at: {doc_path}")

if __name__ == "__main__":
    create_experiment_5_docx()
