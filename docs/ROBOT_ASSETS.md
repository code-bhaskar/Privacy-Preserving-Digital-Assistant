# Contextual robot artwork

## Source

The user-supplied artwork was found on `origin/main` after fetching GitHub, in:

- `robot_gif_frames_named.zip` — fifteen calendar, reminder and summariser frames.
- `robot_modes_separate_named.zip` — Default blue, Privacy green and Global soft-red robots.

Files were **read from those ZIP archives**, not regenerated from prompts. Selected originals were converted to local WebP files in `frontend/public/brand/robots/`. The manifest in that directory records the source commit, original entry names, original-image SHA-256 hashes and ZIP hashes. The original archives remain in GitHub history; they are not duplicated into the application bundle.

The mode PNGs have true transparency. The frame PNGs have an opaque checkerboard baked into their pixels. The import script removes edge-connected light neutral backdrop and narrow crop-neighbour slivers, preserving the enclosed robot materials. It does not invent a new character, recolour the frames or change their camera perspective. Minor baked-in halo detail within holographic rings is retained rather than aggressively deleting character pixels.

## Placement

| Supplied artwork      | Where it appears                                                                                         |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| Default blue robot    | Sign-in illustration, Default-mode welcome, mode picker and ordinary conversation                        |
| Privacy green robot   | Privacy-mode welcome/picker, privacy/settings header                                                     |
| Global soft-red robot | Global-mode welcome/picker and cloud reply identity                                                      |
| Calendar frames       | Calendar suggestion, Calendar workspace, scheduling draft dialog and Clock-mode assistant activity       |
| Reminder frames       | Reminder suggestion, Reminders and Notes workspaces, editing dialogs and Notepad-mode assistant activity |
| Summariser frames     | Summary suggestion, Thinking-mode processing and completed summary reply                                 |

The existing small SVG logo/navigation icons remain for legibility. Large robot artwork is shown uncropped, with transparent backgrounds, in both themes.

## Runtime behavior

`robot-assets.ts` owns the asset mapping and keyword preview classifier. Summary trigger words take precedence over calendar/reminder words inside the document being summarised. The real backend intent replaces the preview once the response arrives; visual keywords never bypass authentication, consent, or confirmation.

`RobotComponent` implements a state-driven frame player:

- **Idle:** the selected blue/green/red mode robot.
- **Context:** a still task-specific calendar/notepad interface for its workspace.
- **Processing:** normal → detect → interface frames; repeat intermediate frames only. No confirmation frame.
- **Review:** transformation → interface; hold a draft illustration. Nothing is saved yet.
- **Success:** the completed frame is allowed only after a successful API result. Fast summaries may finish the decorative thinking/compression sequence after the response is already available; results are never artificially delayed. Return to the selected normal robot after a brief confirmation.
- **Failure:** neutral mode robot and an explicit error label, never a saved/checkmark frame.

Saved Settings reduced-motion preference **and** the OS `prefers-reduced-motion` setting select a still frame instead of cycling. Frame timers stop in hidden tabs and are cleared when the component is destroyed. Image requests are relative/same-origin; no remote image service sees commands.

## GIF sequences

These six-frame standalone reference GIFs are exported with plain dark backgrounds:

- `brand/robots/calendar-sequence.gif`: normal → detection → clock face → calendar → confirmed → normal.
- `brand/robots/reminder-sequence.gif`: normal → detection → notepad → writing → saved → normal.
- `brand/robots/summarizer-sequence.gif`: normal → thinking → deep processing → compression → ready → normal.

**The live UI intentionally plays the individual WebP frames, not an unconditional looping GIF.** That lets it hold on processing/review, respect reduced motion, and show confirmation only when the operation succeeds. The exported GIFs are animation references, not evidence that an operation completed.

## Rebuild and tests

```bash
.venv/bin/pip install -r requirements-assets.txt
.venv/bin/python scripts/prepare_robot_assets.py /path/to/zip-directory <source-commit>
cd frontend
npm run test:assets
npm run build
npm run test:smoke
```

The importer reads only allowlisted archive filenames; it never extracts arbitrary paths. The asset test checks all trigger families, summary precedence, the eighteen WebP assets, three GIFs and the provenance manifest. The browser test exercises all three mode images, intermediate processing frames, no confirmation before save, failed-save/error frames, summary completion, reduced motion and mobile overflow, using an isolated real backend. Optional `PPDA_SCREENSHOT_DIR` captures only the test fixture workspace, not live user data.
