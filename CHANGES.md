## Update: professional "Slate" theme

- `web/theme.css` now holds a calm, professional theme in the style of shadcn/ui and 21st.dev components, replacing the colourful "Aurora" theme. It uses neutral zinc surfaces with one indigo accent, 1px borders, an 8px radius, the Geist font, and light and dark mode that follow the system.
- Tabs, the mode picker and the dialog tabs are segmented controls. Buttons come in fixed 32 and 36 px sizes. Assistant messages have an avatar and no bubble, user messages sit in a muted bubble, the first-run steps form one list card, and toasts are Sonner-style.
- Colour now marks meaning only: green for savings and ready, amber for warnings, red for errors.
- Motion is short (150 to 250 ms) and only shows change: tabs, dialogs, the newest message and the savings bars fade or grow in. The decorative loops are gone. The typing dots, the status pulse and the busy Send shimmer stay, and all of it is off when the system asks for reduced motion.
- The Code Graph's node colours now match the theme.
- Tests: 225 Python and 25 JS pass.

## Update: colourful "Aurora" theme with animations

- New `web/theme.css`, loaded after `styles.css`, recolours the whole app with a violet, pink, teal, amber and sky palette in both light and dark mode. Each tab has its own colour, primary buttons use a gradient, and cards are rounder with soft shadows over a slowly drifting colour background.
- Motion: tabs and panels fade and rise in, the newest chat message pops in, modals spring open, toasts slide in, savings bars grow with a shine, empty-state icons float, and the status dot pulses while ready or indexing. Everything is turned off when the system asks for reduced motion.
- Chat shows an animated "Reading your code and writing an answer..." bubble while waiting for the AI.
- Code Graph: brighter node colours, readable labels in light mode (they were white on white), a colour key in the corner, and a short layout animation.
- Fixed: the folder path field in "Add your code" was very narrow; the floating savings panel on narrow screens was see-through.
- Tests: 225 Python and 25 JS pass.


## Update: upload fix and user-friendly UI

- Upload: failures now report their cause. An unreachable MongoDB returns 503 with a clear message; other 500s include the error type; manager start-up errors are reported. Zips built on Windows (backslash separators) extract correctly, and a single wrapping folder is dropped so imports resolve and the call graph links up.
- Retrieval: with no named symbol, the target is chosen from source files rather than tests.
- UI: guided first run with sample prompts, toasts instead of browser alerts, plain-language labels and tooltips on every tab and dialog, SVG icons instead of emoji, friendlier empty and loading states.
- Fixed: benchmark question-set dropdown never loaded (wrong element id), benchmark results never rendered (summaries arrive keyed by strategy), live-AI checkbox id, stats panel showed zero calls and "SQLite" label.
- Tests: 225 Python and 25 JS pass.
