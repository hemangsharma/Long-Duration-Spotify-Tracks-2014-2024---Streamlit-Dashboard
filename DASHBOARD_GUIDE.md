# Dashboard Guide — Long-Duration Spotify Tracks (2014–2024)

This guide explains what each chart in the Streamlit dashboard shows, how to read it, and what insights to look for.

---

## Dataset (Quick Context)
Each row represents a Spotify track with:
- **Track name** (`name_clean`)
- **Primary artist** (`primary_artist`)
- **Raw artists string** (`artists`)
- **Duration (minutes)** (`duration_minutes`)
- **Artist count** (`artist_count`)
- **Duration tier** (`duration_bucket`): ≤6, 6–10, 10–20, 20–40, 40–60, 60–120, 120+

The dataset is “long-duration” relative to typical songs, but the definition here is “>= ~5 minutes” (min=5, max=100 in the sample).

---

# Filters (Sidebar)

### Duration range (minutes)
Filters the dataset to tracks whose **duration_minutes** fall within the selected range.

### Primary artist
Limits the dataset to selected primary artists. Leave empty to include all.

### Only multi-artist tracks
Shows only tracks where `artist_count > 1`, meaning the raw artist field includes multiple artists.

### Search (track or artist)
Case-insensitive search across:
- Track name (`name_clean`)
- Raw artists (`artists`)
- Primary artist (`primary_artist`)

---

# KPI Cards (Top of Dashboard)

### Tracks
Number of tracks currently shown after filters.

### Unique primary artists
Count of distinct `primary_artist` values in the filtered dataset.

### Median duration (min)
The “typical” duration: half the tracks are shorter, half are longer.

### 90th percentile (min)
A “long track” threshold: only the top 10% are longer than this.

### Max duration (min)
Longest track duration in the current filtered dataset.

---

# Tab: Map Views (Pattern & Hierarchy)

## 1) Content Landscape (Treemap)
**What it shows**
- A “catalog map” where each rectangle area corresponds to total duration.
- Hierarchy: **Primary artist → Track**.
- **Bigger blocks** = longer total duration contribution.

**How to read it**
- Large artist blocks = artists that dominate listening time in your filtered selection.
- Within an artist block, the biggest track rectangles are that artist’s longest tracks.

**What to look for**
- Which artists occupy the most “duration real estate”.
- Whether an artist has many moderately long tracks vs. a few extremely long tracks.

---

## 2) Hierarchy Map (Sunburst): Duration tier → Artist
**What it shows**
- Outer rings break down track counts by **duration tier** and then **artist**.

**How to read it**
- The first ring slices represent tiers (≤6, 6–10, …).
- Inside each tier, artists are sized by how many tracks they have in that tier.

**What to look for**
- Artists who specialize in certain tiers (e.g., mostly 40–60 minutes).
- Whether your filtered dataset skews toward short-long (≤6) or truly long (40+).

---

## 3) Theme Map (from titles): Frequency and typical duration
**What it shows**
- A “topic map” inferred from track titles using curated keyword rules.
- Each theme shows:
  - **Tracks** (count of matched tracks)
  - **Average duration**
  - **Median duration**

**How to read it**
- A bar is long if many titles match that theme’s keywords.
- Hover to see typical duration patterns.

**What to look for**
- Themes that dominate the dataset (e.g., “Nature”, “Meditation”, “Sleep”).
- Themes that correlate with longer durations (e.g., “Singing bowls” might be longer).

**Note**
This is rule-based keyword inference, not NLP/ML. The sidebar lets you customize themes and keywords.

---

## 4) Heatmap: Top artists × Duration tier
**What it shows**
- Rows = top artists (by frequency)
- Columns = duration tiers
- Color intensity = number of tracks

**How to read it**
- Dark cells indicate where an artist has many tracks in a specific tier.

**What to look for**
- Artist “fingerprints” across tiers.
- Specialization (one tier dominates) vs. diversity (tracks spread across tiers).

---

# Tab: Trends

## 5) Duration Distribution (Histogram + Box)
**What it shows**
- Histogram: how many tracks fall into each duration range.
- Box plot (marginal): median, spread, and outliers.

**How to read it**
- Peaks = most common durations.
- The boxplot reveals outliers and skew.

**What to look for**
- Whether the dataset clusters heavily around 5–10 minutes.
- Whether the distribution has a heavy tail (many longer tracks).

---

## 6) Track Duration Tiers (Donut)
**What it shows**
- Proportion of tracks in each `duration_bucket`.

**How to read it**
- Bigger donut slices = more tracks in that tier.

**What to look for**
- How much of the dataset is only slightly long (≤6 / 6–10)
  vs. truly long (20+).

---

## 7) Top Artists (Bar)
**What it shows**
- Top-N artists by track count.

**How to read it**
- Longer bars = more tracks.

**What to look for**
- Dominant artists and whether filters change the leaders.

---

## 8) Longest Tracks (Table)
**What it shows**
- Top 20 tracks by duration within current filters.

**How to use it**
- Use this to inspect extremes and validate whether the “long track” label fits your use case.

---

# Tab: Data Explorer

## 9) Browse Tracks (Interactive Table)
**What it shows**
- Filtered dataset in a sortable table.

**Tips**
- Use search + filters to narrow to a theme.
- Use sorting to quickly find the longest tracks or track groups by artist.

---

## 10) Download Filtered Dataset (CSV)
Exports the currently filtered dataset to CSV, useful for:
- further analysis
- sharing selections
- reproducibility for portfolio artifacts

---

# Suggested Portfolio Narratives (Examples)

### Narrative A: “Long-form audio is mostly short-long”
Use histogram + donut to show most tracks sit in ≤6 or 6–10 minutes, with a smaller tail.

### Narrative B: “Some artists dominate listening time”
Use treemap to show which artists occupy the most total duration.

### Narrative C: “Themes have duration signatures”
Use theme map + heatmap to show that themes like “Sleep” or “Meditation” tend to have different typical durations.

---