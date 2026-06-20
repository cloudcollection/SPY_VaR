# Section 2 plotting style notes

The figures in Section 2 use a static academic-publication workflow:

- Matplotlib is the rendering engine because it is already installed in the project environment.
- The style system follows a restrained grammar-of-graphics approach associated with popular Python/R plotting libraries: fixed semantic tokens, explicit palette maps, quiet grids, visible axis anchors, and neutral descriptive titles/subtitles.
- PNG files are exported at 320 dpi for direct insertion into the thesis; SVG files are exported alongside them for later vector editing.
- The palette is intentionally limited to blue, gold, orange, olive, pink, and neutrals. This avoids default rainbow colors and keeps the chapter visually consistent.
- Every figure states the metric scope in the subtitle so the chart remains interpretable after being copied into Word, LaTeX, or PowerPoint.
- Figure titles are descriptive rather than promotional or conclusion-led; substantive interpretation belongs in the surrounding thesis text.
