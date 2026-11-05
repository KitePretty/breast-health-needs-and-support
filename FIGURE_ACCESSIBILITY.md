# Figure appearance and accessibility

The shared plotting code uses the palette of the local submission Figure 2 (blue, gold, and red) and Figure 3 (cividis). The earlier auxiliary red–yellow–green heatmap has been replaced. The source submission PNG files and statistical results are not modified.

## Encodings

| Measure | Color | Shape |
|---|---|---|
| Receipt of a human reply | Blue `#24688F` | Circle |
| Target category among analyzable responding threads | Gold `#D18C20` | Square |
| Overall support-category coverage | Red `#B64B45` | Diamond |

All three marker types have a dark outline and printed numeric labels. Color is supplementary to shape and position. The gold fill has lower contrast on white, so its dark boundary is important. Heatmaps use a common 0–100% cividis scale with printed values. Missing/ineligible cells have a hatch and dash and are not represented as zero. Black or white heatmap label text is selected from the cell background's relative luminance, choosing the higher contrast.

The numerical checks use 4.5:1 as the ordinary-text contrast target and 3:1 for marker boundaries. These are practical design targets informed by the [W3C text contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html) and [non-text contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html). The checks cover selected color pairs, not a full WCAG conformance audit. No color-vision-deficiency simulation or assistive-technology user study is claimed.

## Output files

Run `python src/render_figures.py` after the aggregate tables have been calculated. It writes:

- PNG at 300 dpi, plus SVG with searchable text and PDF with embedded TrueType fonts.
- `Figure2_data.csv` and `Figure3_data.csv`, containing the displayed aggregate values at full numeric precision.
- `alt_text.json`, containing the two main figure descriptions below.
- `accessibility_checks.json`, containing numerical contrast checks.
- `grayscale_preview/`, containing plots rendered without hue to check marker/label readability.

The PNG is a raster export. The SVG/PDF and adjacent data table provide additional ways to inspect the plotted information. Do not assume that an adjacent JSON description is automatically attached to an uploaded image. Add the alt text when inserting figures into the manuscript or repository page.

## Main manuscript Figure 2: suggested alt text

Dot plot of five need–support mappings. Circles show receipt of a human reply, squares show target support-category presence among threads with analyzable replies, and diamonds show overall support-category coverage. Every marker has a printed percentage. Overall coverage ranges from 16.2% for risk assessment to help-seeking encouragement to 73.8% for medical-process information to informational support. The three measures use different denominators. Connecting lines group estimates and do not represent change over time.

## Main manuscript Figure 3: suggested alt text

Heatmap of overall informational-support coverage across seven topics within the Diagnostic position, with columns for symptom interpretation, risk assessment, and medical-process information needs. Every eligible cell has a percentage. Reading Clinical Reports has 13%, 13%, and 18% coverage; Waiting for Results has 84%, 76%, and 90%. Hatched cells with a dash did not meet eligibility thresholds and are not zero coverage. The color scale runs from dark blue at low coverage to yellow at high coverage.

## Auxiliary figures

- `figure1_study_and_analysis_flow`: left-to-right context diagram from the 15,037-thread topic corpus to the 1,000-thread coded sample, response access, and support-category coverage. LDA is contextual information and is not rerun by this package.
- `figure2_c1_communication_heatmap`: twelve post-feature rows by four pathway positions, with a common 0–100% scale and printed percentages.
- `figure3_two_stage_support_pathway`: the same encoding, layout, and data as main manuscript Figure 2.
- `figure4_c1_support_heatmap`: five thread-level support-category rows by four pathway positions, including unanswered threads and excluding unknown response content.
- `figure5_c1_3_topic_information_gaps`: the same encoding, layout, and data as main manuscript Figure 3.

Auxiliary figure numbers are internal filenames and differ from the main manuscript numbering. For submission, use `outputs/manuscript_figures/`.

## Journal preparation

[JAMIA's author instructions](https://academic.oup.com/jamia/pages/General_Instructions) request alt text for main-article figures, placed below the relevant caption in the main manuscript and preceded by `Alt text:`. The descriptions here are suggestions for that use; this package does not edit the manuscript.
