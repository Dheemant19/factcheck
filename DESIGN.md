---
name: Fact-Check Lab
description: Paste a claim, get an explained verdict, set like a published fact-check.
colors:
  prussian-blue: "#1e3a5f"
  paper: "#faf8f5"
  stone: "#f1ede6"
  stone-deep: "#e8e2d8"
  ink: "#18181b"
  muted-ink: "#52525b"
  hairline: "#d6d1c7"
  unlit-rung: "#ebe6dd"
  verdict-brick: "#b8332a"
  verdict-brick-text: "#9e2b23"
  verdict-brick-tint: "#f8e3e0"
  verdict-brick-deep: "#8a241d"
  verdict-rust: "#a04a1c"
  verdict-ochre: "#9a5b00"
  verdict-ochre-text: "#8a5200"
  verdict-ochre-tint: "#f7ead0"
  verdict-moss: "#3f6e3a"
  verdict-green: "#0b6e4f"
  verdict-green-text: "#0a5f44"
  verdict-green-tint: "#dcefe6"
  shortcut-tint: "#dde7f3"
  results-bar: "#3a5a80"
typography:
  masthead:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "2.6rem"
    fontWeight: 600
  ruling:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "2.1rem"
    fontWeight: 600
  section:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "1.35rem"
    fontWeight: 600
  column-title:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "1.15rem"
    fontWeight: 600
  body:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
  chart-label:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
  code:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontWeight: 400
rounded:
  base: "2px"
spacing:
  page-top: "1.5rem"
components:
  button-primary:
    backgroundColor: "{colors.prussian-blue}"
    textColor: "{colors.paper}"
    rounded: "{rounded.base}"
  button-tertiary:
    textColor: "{colors.ink}"
  input:
    backgroundColor: "{colors.stone}"
    textColor: "{colors.ink}"
    rounded: "{rounded.base}"
  view-switch-active:
    textColor: "{colors.prussian-blue}"
    rounded: "{rounded.base}"
  verdict-block:
    backgroundColor: "{colors.paper}"
    rounded: "{rounded.base}"
  ladder-rung-unlit:
    backgroundColor: "{colors.unlit-rung}"
    textColor: "{colors.muted-ink}"
    rounded: "{rounded.base}"
    height: "36px"
  ladder-rung-lit-false:
    backgroundColor: "{colors.verdict-brick-tint}"
    textColor: "{colors.verdict-brick-text}"
    rounded: "{rounded.base}"
    height: "36px"
  ladder-rung-lit-true:
    backgroundColor: "{colors.verdict-green-tint}"
    textColor: "{colors.verdict-green-text}"
    rounded: "{rounded.base}"
    height: "36px"
---

# Design System: Fact-Check Lab

## Overview

**Creative North Star: "The Broadsheet Dossier"**

A verdict reads like a published fact-check: the ruling set in a serif line, then how sure the model is, then the words behind it, then the evidence. The page is warm paper with charcoal ink, one Prussian-blue action colour, and a small set of verdict colours that carry meaning and nothing else. It refuses the dashboard default the first build had: navy chrome, pill badges and rows of same-size cards.

The whole system is the native Streamlit theme (`app/.streamlit/config.toml`) plus native components; there is no custom CSS, and that is a confirmed constraint, not a gap. Everything the theme cannot hold (the rating ladder, the push bars, the Results ladder) is drawn in Altair with colours named once at the top of `app/app.py`. The app is light only and is built for a 1080p screen recording: every chart value is printed on the chart, nothing depends on hover.

Density is editorial rather than dashboard: wide layout, generous gaps between sections, few borders, and hierarchy carried by the serif sizes instead of boxes.

**Key Characteristics:**
- Warm paper ground, stone for inputs and the sidebar, hairline rules.
- One action colour (Prussian blue); verdict colours mean verdicts and word pushes only.
- Newsreader serif for every heading, with the 2.1rem size reserved for the ruling.
- 2px corners everywhere; borders only around the verdict, the second opinion and the claim exhibits.
- Every number is labelled on screen; no hover-only values, no 100% or 0%.

## Colors

A warm, low-chroma paper palette with one cool action blue and a brick / ochre / bottle-green verdict triad.

### Primary
- **Prussian Blue** (prussian-blue): the only action colour. The primary Check button, links (underlined), the active view in the segmented switch, focus. Also the darkest step of the chart categorical ramp.

### Secondary
- **Results Slate Blue** (results-bar): the single hue of the Results ladder-of-models bars. A lighter step of the Prussian ramp, so scores never borrow a verdict colour.
- **Shortcut Tint** (shortcut-tint): theme blue background. Marks a style shortcut (the "WASHINGTON (Reuters) -" dateline) in the Shortcut demo and nothing else.

### Tertiary: verdict colours
- **Brick** (verdict-brick, text verdict-brick-text, tint verdict-brick-tint): Likely false, false-side ladder rungs, words and bars pushing towards false. Deepened to verdict-brick-deep for "Pants on fire" and warmed to verdict-rust for "Barely true".
- **Ochre** (verdict-ochre, text verdict-ochre-text, tint verdict-ochre-tint): Mixed / uncertain and the middle rungs (half-true, mixture, disputed). The theme's yellow is set to the same ochre so no second warning colour exists.
- **Bottle Green** (verdict-green, text verdict-green-text, tint verdict-green-tint): Likely true, true-side rungs, pushes towards true. Mossed to verdict-moss for "Mostly true".

The text variants pass WCAG AA on paper and on their own tint; the rung fills are dark enough for white text.

### Neutral
- **Paper** (paper): page ground and the background of the verdict block.
- **Stone** (stone): text areas, selects, code blocks, dataframe headers, the sidebar. **Stone Deep** (stone-deep) is the sidebar's secondary background.
- **Ink** (ink): body text and the outline of a lit rung when the verdict is not confident.
- **Muted Ink** (muted-ink): captions, plates, metadata, unlit-rung labels, grey status words.
- **Hairline** (hairline): widget borders, dividers, the sidebar edge, dataframe borders.
- **Unlit Rung** (unlit-rung): the theme's grey background; unlit ladder rungs and the "no claim yet" state.

### Named Rules
**The Verdicts-Only Rule.** Brick, ochre and green mean a verdict or a word's push towards one. They never decorate, never mark status chrome, and never colour a model family in a chart.

**The One Action Rule.** Prussian blue is the only colour that asks to be pressed. Secondary actions are tertiary buttons or toggles in ink.

**The Blue Tint Means Style Rule.** The blue background marks a style shortcut, the source of a model's cheat; it is never used as a generic highlight.

## Typography

**Display Font:** Newsreader (with Georgia, serif), optical sizes 6..72, weights 500-700 loaded
**Body Font:** Public Sans (with system-ui, sans-serif), weights 400-700 loaded
**Label/Mono Font:** IBM Plex Mono, for code and commands only

**Character:** A newspaper serif for every heading against a plain civic sans for reading; the pairing says "published ruling" rather than "control panel".

### Hierarchy
- **Masthead** (600, 2.6rem): "Fact-Check Lab" only, as the page title.
- **Ruling** (600, 2.1rem): the verdict line inside the verdict block, coloured by verdict with its icon. Reserved; no other h2 appears.
- **Section** (600, 1.35rem): Why, Evidence, the Results section titles, the verdict in a Compare column.
- **Column title** (600, 1.15rem): a model's name above its why column, a dataset name above its chart, the second-opinion title.
- **Body** (400, 17px): reading lines, claims, evidence text. Bold for the probability and the most likely label.
- **Caption** (theme caption size, muted ink): model plates, timing lines, method lines, similarity scores.
- **Chart labels** (14px word and rung labels, 13px values and axis titles in normal weight, 12px ticks).

### Named Rules
**The Reserved Ruling Rule.** The 2.1rem serif size belongs to the verdict. If something else needs weight, use the 1.35rem section size.

**The Sentence Case Rule.** Verdicts and labels are sentence case ("Likely false", "Mixed / uncertain", "Pants on fire"); view names are plain ("Check a claim", "Compare models", "Shortcut demo", "Results").

## Layout

Wide layout with the sidebar collapsed; page top padding is the page-top spacing. A masthead row puts the title and a one-line muted description at left and the four-view segmented switch at right, bottom-aligned, then a divider.

Check: the claim box and its example links take a 6.4 column against a 3.6 side column holding the model select, the second-opinion toggle and a full-width Check button. Below sit the verdict block, the optional second-opinion block, the Why section as two equal columns (TF-IDF and the transformer, on a shared push scale), and Evidence as two columns. Compare puts up to four model columns side by side; Results sets the ladder charts two per row, then the specs board and the full runs table. Columns use Streamlit's "large" gap between peers and "medium" inside control rows; sections are separated by native "medium" or "large" space. A divider and a muted status caption close every view.

## Elevation & Depth

Flat. There are no shadows; depth is tonal (stone inputs on paper) and structural (hairline borders and dividers). The only figure-ground move is the lit rung: a pale verdict tint inside a 2px verdict-colour outline.

### Named Rules
**The Flat Paper Rule.** Nothing floats. A surface is set apart by a hairline or a tint, never a shadow.

## Shapes

Barely softened corners (2px) on widgets, buttons, bordered containers, ladder rungs and chart bars. Borders are reserved: the verdict block, the second-opinion block, and the claim exhibits (the tinted claim in a why or Compare column, the article in the Shortcut demo) are the only bordered containers, and they never nest.

### Named Rules
**The Three Frames Rule.** Only rulings and exhibits get a frame. Sections, lists and charts sit on open paper.

## Components

### Buttons
- **Shape:** 2px corners.
- **Primary:** Prussian blue fill with paper text and a material icon (Check, with the fact-check icon), full width in its column. One per view at most.
- **Tertiary:** example claims render as quiet text buttons in ink after a muted "Examples" caption; one click loads and checks. The full claim is in the tooltip, the short name is on the button.
- **Tertiary with icon:** incidental actions stay tertiary too (the Shortcut demo's "Another article" with the shuffle icon); no secondary buttons are used.

### Inputs / Fields
- **Style:** stone fill, hairline widget border, 2px corners; the claim box is 118px tall with a placeholder ("Paste a claim, a post or a headline…").
- **Focus:** native Streamlit focus in Prussian blue.
- **Disabled:** native disabled state when no models exist, with a designed empty panel explaining what fills it.

### Navigation
- **Style:** a native segmented control of the four plain view names at the right of the masthead; the active view is drawn in Prussian blue. The view is mirrored to `?view=` so a link opens it.

### Verdict Block (signature)
A bordered paper container: the ruling in the 2.1rem serif, verdict-coloured with its material icon (check circle, help, cancel); a body line with the probability in bold ("**>99%** probability false", plus the most likely label when the dataset has more than two); a 36px rating ladder of rungs from false to true, each reading "name  probability", with "← false" and "true →" captions under its ends; then the model plate and timing line as a caption. The lit rung is outlined in its verdict colour and tinted only when the verdict is confident, so a weak top label never looks like a ruling. Before a check, the ruling reads "No claim checked yet" in grey.

### Paired Why (signature)
Two equal columns, one per model, each: the model name, a verdict-and-probability line, the claim in a bordered exhibit with its eight strongest words on a verdict tint (the two strongest bold), then horizontal push bars (brick towards false, green towards true, 15px, 2px corners) with each value printed beyond the bar end, on an x scale shared by both columns so equal pushes draw equal bars. A method caption names the explainer and its time.

### Evidence List
Two columns: similar already-fact-checked claims led by a bold verdict-coloured label, with a muted similarity score; and the closest evidence sentences with a muted source and score. A caption states how retrieval relates to the verdict.

### Results Ladder
Small multiples, two per row, one per dataset: a horizontal bar per rung in the single Results slate blue, bottom rung on top, each labelled with its score and best run name; a fixed 0 to 1 axis titled "Test macro-F1, mean over seeds".

### Specs Board
A column grid (ID, Specification, Target, Best, Status) with muted caption headers; status is an icon plus a word: plain "Pass", bold brick "Fail", grey "No runs yet" and "Check by hand".

### Footer
A divider and a single muted caption line: the product's positioning, the models found, the device and live GPU memory.

## Do's and Don'ts

### Do:
- **Do** take every colour, font and size from `app/.streamlit/config.toml` and the named constants in `app/app.py` (RUNG_FILL, RUNG_TINT, RUNG_INK, PUSH_FALSE, PUSH_TRUE, the Results bar colour).
- **Do** print every chart value on the chart; the app is read from a 1080p video frame.
- **Do** print probabilities as ">99%" and "<1%" at the extremes, never 100% or 0%.
- **Do** put the reason and the evidence next to every verdict.
- **Do** give unavailable panels a designed empty state that says what will appear and what fills it.

### Don't:
- **Don't** add custom CSS or HTML styling; native theme and components only.
- **Don't** use brick, ochre or green for anything but verdicts and word pushes, or the blue tint for anything but style shortcuts.
- **Don't** add pill badges, eyebrows or kickers above headings, or numbered section labels.
- **Don't** nest bordered containers or frame sections, lists and charts.
- **Don't** add a dark mode; the theme is locked to light so the recording looks the same every time.
