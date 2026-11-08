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
  zero-line: "#a8a29e"
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
    fontWeight: 700
  column-title:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "1.15rem"
    fontWeight: 600
  body:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
  push-word:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
  push-value:
    fontFamily: "IBM Plex Mono, monospace"
    fontSize: "0.88rem"
    fontWeight: 400
    fontFeature: "tnum"
  small-muted:
    fontFamily: "Public Sans, system-ui, sans-serif"
    fontSize: "0.88rem"
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
  push-bar-false:
    backgroundColor: "{colors.verdict-brick}"
    rounded: "{rounded.base}"
    height: "0.95rem"
  push-bar-true:
    backgroundColor: "{colors.verdict-green}"
    rounded: "{rounded.base}"
    height: "0.95rem"
---

# Design System: Fact-Check Lab

## Overview

**Creative North Star: "The Broadsheet Dossier"**

A verdict reads like a published fact-check: the ruling set in a serif line, then one sentence on how sure the model is, then the words behind it, then the evidence. The page is warm paper with charcoal ink, one Prussian-blue action colour, and a small set of verdict colours that carry meaning and nothing else. It refuses the dashboard default the first build had: navy chrome, pill badges and rows of same-size cards.

The system is the native Streamlit theme (`app/.streamlit/config.toml`) plus native components, with one small stylesheet (the `STYLE` constant in `app/app.py`, injected once with `st.html`) for the few things the theme cannot do: the word-push rows, heavier section headings, hidden heading link icons, the selection tint and the muted source line. The thin rating ladder and the Results ladder-of-models are drawn in Altair. Colours are named once, in the theme and at the top of `app/app.py`. The app is light only and is built for a 1080p screen recording: every value is printed on screen, nothing depends on hover.

The page shows only what a person checking a claim needs. Copy is full sentences with the key fact in bold, never fragments joined by separators. Model plates, seeds, test scores, timings, spec IDs and method notes are kept out of the checking views and live only in Results. Density is editorial rather than dashboard: wide layout, generous gaps between sections, few borders, and hierarchy carried by the serif sizes instead of boxes.

**Key Characteristics:**
- Warm paper ground, stone for inputs and the sidebar, hairline rules.
- One action colour (Prussian blue); verdict colours mean verdicts and word pushes only.
- Newsreader serif for every heading, with the 2.1rem size reserved for the ruling.
- 2px corners everywhere; borders only around the verdict, the second opinion and the claim exhibits.
- Full sentences with the key fact in bold; no middle-dot strings, no run metadata next to a verdict.
- Every number is labelled on screen; no hover-only values, no 100% or 0%.

## Colors

A warm, low-chroma paper palette with one cool action blue and a brick / ochre / bottle-green verdict triad.

### Primary
- **Prussian Blue** (prussian-blue): the only action colour. The primary Check button, links (underlined), the active view in the segmented switch, focus. Also the darkest step of the chart categorical ramp.

### Secondary
- **Results Slate Blue** (results-bar): the single hue of the Results ladder-of-models bars. A lighter step of the Prussian ramp, so scores never borrow a verdict colour.
- **Shortcut Tint** (shortcut-tint): theme blue background. Marks a style shortcut (the "WASHINGTON (Reuters) -" dateline) in the Shortcut demo; the same pale blue is the text-selection tint.

### Tertiary: verdict colours
- **Brick** (verdict-brick, text verdict-brick-text, tint verdict-brick-tint): Likely false, false-side ladder rungs, words and push bars towards false. Deepened to verdict-brick-deep for "Pants on fire" and warmed to verdict-rust for "Barely true".
- **Ochre** (verdict-ochre, text verdict-ochre-text, tint verdict-ochre-tint): Mixed / uncertain and the middle rungs (half-true, mixture, disputed). The theme's yellow is set to the same ochre so no second warning colour exists.
- **Bottle Green** (verdict-green, text verdict-green-text, tint verdict-green-tint): Likely true, true-side rungs, pushes towards true. Mossed to verdict-moss for "Mostly true".

The text variants pass WCAG AA on paper and on their own tint; the rung fills are dark enough for white text.

### Neutral
- **Paper** (paper): page ground and the background of the verdict block.
- **Stone** (stone): text areas, selects, code blocks, dataframe headers, the sidebar. **Stone Deep** (stone-deep) is the sidebar's secondary background.
- **Ink** (ink): body text and the outline of a lit rung when the verdict is not confident.
- **Muted Ink** (muted-ink): the "Try" label, the push key, evidence source lines, unlit-rung labels, specs headers, grey status words.
- **Hairline** (hairline): widget borders, dividers, the sidebar edge, dataframe borders.
- **Zero Line** (zero-line): the 1px vertical zero rule the push bars grow from; a step darker than hairline so it reads across the rows.
- **Unlit Rung** (unlit-rung): the theme's grey background; unlit ladder rungs and the "no claim yet" state.

### Named Rules
**The Verdicts-Only Rule.** Brick, ochre and green mean a verdict or a word's push towards one. They never decorate, never mark status chrome, and never colour a model family in a chart.

**The One Action Rule.** Prussian blue is the only colour that asks to be pressed. Secondary actions are tertiary buttons or toggles in ink.

**The Blue Tint Means Style Rule.** In content, the blue background marks a style shortcut, the source of a model's cheat; it is never used as a generic highlight. Its only other use is the text-selection tint.

## Typography

**Display Font:** Newsreader (with Georgia, serif), optical sizes 6..72, weights 500-700 loaded
**Body Font:** Public Sans (with system-ui, sans-serif), weights 400-700 loaded
**Label/Mono Font:** IBM Plex Mono, for the push values (tabular figures) and for code and commands

**Character:** A newspaper serif for every heading against a plain civic sans for reading; the pairing says "published ruling" rather than "control panel". The mono appears only where digits must line up.

### Hierarchy
- **Masthead** (600, 2.6rem): "Fact-Check Lab" only, as the page title. Nothing sits under it.
- **Ruling** (600, 2.1rem): the verdict line inside the verdict block, coloured by verdict with its icon. Reserved; no other h2 appears.
- **Section** (700, 1.35rem): Why, Evidence, the Shortcut demo's "Trained on raw text" / "Trained on cleaned text", the Results section titles. A step heavier than the theme's 600, set in the stylesheet.
- **Column title** (600, 1.15rem): a model's name above its why or Compare column, a dataset name above its Results chart, the second-opinion title.
- **Body** (400, 17px): reading lines, claims, evidence text, verdict sentences. Bold carries the verdict word, the probability and the most likely label.
- **Push word** (400, 1rem; 700 for the two strongest): the word column of the push rows, right-aligned and truncated with an ellipsis.
- **Push value** (IBM Plex Mono 400, 0.88rem, tabular figures): signed values ("+0.89", "-0.01") in their own right-aligned column.
- **Small muted** (400, 0.88rem, muted ink): the "From a PUBHEALTH fact-check article" source line; the push key ("towards true", "towards false") is 0.85rem.
- **Caption** (theme caption size, muted ink): the "Try" label, the false/true arrows under ladders of four or more rungs, specs headers, "No single word moved this verdict."
- **Chart labels** (14px rung and Results axis labels, 13px Results values and axis title in normal weight).

### Named Rules
**The Reserved Ruling Rule.** The 2.1rem serif size belongs to the verdict. If something else needs weight, use the 1.35rem section size.

**The Sentence Case Rule.** Verdicts and labels are sentence case ("Likely false", "Mixed / uncertain", "Pants on fire"); view names are plain ("Check a claim", "Compare models", "Shortcut demo", "Results").

**The Full Sentence Rule.** Every line the reader reads is a full sentence with the key fact in bold: "**Likely false**, with a **>99%** probability that the claim is false." In the verdict block, where the ruling is already the heading: "The model gives a **>99%** probability that this claim is false." Never join phrases with a middle dot (·) or similar separators anywhere in the app.

**The Checker's Eye Rule.** Show only what a person checking a claim needs. No model plates, seeds, test F1 scores, timings, spec IDs or method captions next to a verdict, and no explanatory captions under charts; run details live only in the Results view.

## Layout

Wide layout with the sidebar collapsed; page top padding is the page-top spacing. A masthead row puts the title at left and the four-view segmented switch at right, centred on one line, then a divider. There is no standfirst under the title and no footer.

Check: the claim box with the "Try" example row under it takes a 6.4 column against a 3.6 side column holding "Trained on" (the dataset in plain words), "Model" (plain model names), the second-opinion toggle and a full-width Check button. Below sit the verdict block, the optional second-opinion block, the Why section as two equal columns (the TF-IDF baseline and its transformer counterpart, left to right up the ladder, on a shared push scale), and Evidence as two columns. Compare uses the same control row with a "Models" multiselect (up to four) and puts the model columns side by side. The Shortcut demo is one sentence of context, a toggle and a tertiary button, the bordered article, then raw and cleaned columns. Results sets the ladder charts two per row, then the specs board and the full runs table. Columns use Streamlit's "large" gap between peers and "medium" inside control rows; sections are separated by native "small" or "medium" space.

## Elevation & Depth

Flat. There are no shadows; depth is tonal (stone inputs on paper) and structural (hairline borders and dividers). The only figure-ground move is the lit rung: a pale verdict tint inside a 2px verdict-colour outline.

### Named Rules
**The Flat Paper Rule.** Nothing floats. A surface is set apart by a hairline or a tint, never a shadow.

## Shapes

Barely softened corners (2px) on widgets, buttons, bordered containers, ladder rungs, push bars and chart bars. Borders are reserved: the verdict block, the second-opinion block, and the claim exhibits (the tinted claim in a why or Compare column, the article in the Shortcut demo) are the only bordered containers, and they never nest.

### Named Rules
**The Three Frames Rule.** Only rulings and exhibits get a frame. Sections, lists and charts sit on open paper.

## Components

### Buttons
- **Shape:** 2px corners.
- **Primary:** Prussian blue fill with paper text and a material icon (Check, with the fact-check icon), full width in its column. One per view at most.
- **Tertiary:** example claims are quiet text buttons in ink after a muted "Try" caption; one click loads the claim, sets the dataset that claim belongs to, and checks it.
- **Tertiary with icon:** incidental actions stay tertiary too (the Shortcut demo's "Another article" with the shuffle icon); no secondary buttons are used.

### Inputs / Fields
- **Style:** stone fill, hairline widget border, 2px corners; the claim box ("Claim or post") is 118px tall with a placeholder ("Paste a claim, a post or a headline…").
- **Model choice:** two selects. "Trained on" names the dataset in plain words with its short name in parentheses ("Political statements (LIAR)"); "Model" lists plain model names, one per run ("RoBERTa-base on cleaned text", "TF-IDF + logistic regression"). The dataset choice is shared across views.
- **Focus:** native Streamlit focus in Prussian blue.
- **Disabled:** native disabled state when no models exist, with a designed empty panel explaining what fills it.

### Navigation
- **Style:** a native segmented control of the four plain view names at the right of the masthead; the active view is drawn in Prussian blue. The view is mirrored to `?view=` so a link opens it.

### Verdict Block (signature)
A bordered paper container: the ruling in the 2.1rem serif, verdict-coloured with its material icon (check circle, help, cancel); one sentence with the probability in bold, naming the most likely label when the dataset has more than two; then a 36px rating ladder of rungs from false to true, each reading "name  probability". Two- and three-rung ladders carry no direction captions (the rung names say it); ladders of four or more get muted "← false" and "true →" captions under their ends. The lit rung is outlined in its verdict colour and tinted only when the verdict is confident, so a weak top label never looks like a ruling. Before a check, the ruling reads "No claim checked yet" in grey with one sentence on how to start. Nothing else sits in the block.

### Word-Push Rows (signature)
The strongest pushes as aligned HTML rows on a three-column grid: the word (right-aligned, the two strongest bold), a track with a bar growing from a 1px zero line, and the signed value in its own IBM Plex Mono tabular column. A muted key row above reads "towards true" at the left end and "towards false" at the right. Brick bars push towards false, green towards true, 2px corners. Columns that sit side by side share one scale so equal pushes draw equal bars. Eight rows in Check, five in Compare, four in the Shortcut demo.

### Paired Why
Two equal columns, one per model: the model name as a column title, its verdict sentence, the claim in a bordered exhibit with its eight strongest words on a verdict tint (the two strongest bold), then the word-push rows. No legend or method caption.

### Evidence List
Two columns under bold run-in titles: similar already-fact-checked claims, each led by a bold verdict-coloured label; and the closest evidence sentences, each followed by a muted source line ("From a PUBHEALTH fact-check article"). No similarity scores and no retrieval caption.

### Results Ladder
Small multiples, two per row, one per dataset: a horizontal bar per rung in the single Results slate blue, bottom rung on top, each labelled with its score and best run name; a fixed 0 to 1 axis titled "Test macro-F1, mean over seeds".

### Specs Board
Results only. A column grid (ID, Specification, Target, Best, Status) with muted caption headers; status is an icon plus a word: plain "Pass", bold brick "Fail", grey "No runs yet" and "Check by hand".

## Do's and Don'ts

### Do:
- **Do** take every colour, font and size from `app/.streamlit/config.toml`, the named constants in `app/app.py` (RUNG_FILL, RUNG_TINT, RUNG_INK, the Results bar colour) and the `STYLE` stylesheet.
- **Do** keep CSS to the one small `STYLE` stylesheet, and only for what the theme cannot do (the push rows, section weight, heading link icons, selection tint, the source line).
- **Do** write full sentences with the key fact in bold: "**Likely false**, with a **>99%** probability that the claim is false."
- **Do** print every value on screen; the app is read from a 1080p video frame.
- **Do** print probabilities as ">99%" and "<1%" at the extremes, never 100% or 0%.
- **Do** name datasets and models in plain words ("Political statements (LIAR)", "RoBERTa-base on cleaned text").
- **Do** put the reason and the evidence next to every verdict.
- **Do** give unavailable panels a designed empty state that says what will appear and what fills it.

### Don't:
- **Don't** join phrases with a middle dot (·) anywhere in the app.
- **Don't** put model plates, seeds, test F1 scores, timings, spec IDs or method captions next to a verdict; run details belong in Results.
- **Don't** add standfirsts, footers, status lines, legend captions or loading notes that a person checking a claim does not need.
- **Don't** grow the stylesheet into a second theme, or restyle what the native theme already handles.
- **Don't** use brick, ochre or green for anything but verdicts and word pushes, or the blue tint in content for anything but style shortcuts.
- **Don't** add pill badges, eyebrows or kickers above headings, or numbered section labels.
- **Don't** nest bordered containers or frame sections, lists and charts.
- **Don't** add a dark mode; the theme is locked to light so the recording looks the same every time.
