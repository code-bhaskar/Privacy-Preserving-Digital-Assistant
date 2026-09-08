import {
  ChangeDetectorRef,
  Component,
  Input,
  OnChanges,
  OnDestroy,
} from "@angular/core";
import {
  modeRobot,
  ROBOT_FRAMES,
  RobotIntent,
  RobotPhase,
} from "./robot-assets";

/** Frame player driven by real operation state, not a looping success GIF.
 * Processing excludes frame 5. Only an explicit successful API result unlocks it.
 */
@Component({
  selector: "app-robot",
  standalone: true,
  template: `<figure
    class="robot-character"
    [class]="'robot-character robot-' + size"
    [class.robot-error]="phase === 'error'"
    [attr.data-robot-intent]="intent"
    [attr.data-robot-phase]="phase"
  >
    <div class="robot-stage">
      <span class="robot-aura" aria-hidden="true"></span
      ><img
        class="robot-art"
        [src]="source"
        [alt]="description"
        width="307"
        height="341"
        decoding="async"
        (error)="fallback()"
      />
    </div>
    @if (label) {
      <figcaption><span class="dot"></span>{{ label }}</figcaption>
    }
  </figure>`,
  styles: [
    `
      :host {
        display: block;
        flex-shrink: 0;
      }
      .robot-character {
        position: relative;
        margin: 0;
        text-align: center;
        width: 210px;
        --glow: #75bde52c;
      }
      .robot-stage {
        position: relative;
        display: grid;
        place-items: center;
        height: 218px;
      }
      .robot-art {
        position: relative;
        z-index: 1;
        width: 100%;
        height: 100%;
        max-height: 218px;
        object-fit: contain;
        filter: drop-shadow(0 12px 16px #15324a15);
      }
      .robot-aura {
        position: absolute;
        width: 145px;
        height: 145px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--glow), transparent 70%);
        filter: blur(8px);
      }
      figcaption {
        position: relative;
        z-index: 2;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: -5px;
        padding: 7px 12px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--surface);
        font-size: 10px;
        color: var(--muted);
        white-space: nowrap;
      }
      .robot-compact {
        width: 98px;
      }
      .robot-compact .robot-stage,
      .robot-compact .robot-art {
        height: 108px;
        max-height: 108px;
      }
      .robot-compact .robot-aura {
        width: 75px;
        height: 75px;
      }
      .robot-panel {
        width: 135px;
      }
      .robot-panel .robot-stage,
      .robot-panel .robot-art {
        height: 145px;
        max-height: 145px;
      }
      .robot-panel figcaption {
        font-size: 9px;
      }
      .robot-panel .robot-aura {
        width: 105px;
        height: 105px;
      }
      .robot-error {
        --glow: #dfac6833;
      }
      .robot-error .dot {
        background: #c9a16e;
      }
      [data-robot-intent="reminder"] {
        --glow: #e8c4742d;
      }
      [data-robot-intent="summarizer"] {
        --glow: #ad98ee2d;
      }
      @media (max-width: 760px) {
        .robot-character.robot-hero {
          width: 185px;
        }
        .robot-hero .robot-stage,
        .robot-hero .robot-art {
          height: 185px;
          max-height: 185px;
        }
        .robot-panel {
          width: 100px;
        }
        .robot-panel .robot-stage,
        .robot-panel .robot-art {
          height: 112px;
          max-height: 112px;
        }
      }
    `,
  ],
})
export class RobotComponent implements OnChanges, OnDestroy {
  @Input() mode = "Default";
  @Input() intent: RobotIntent = "idle";
  @Input() phase: RobotPhase = "idle";
  @Input() reduced = false;
  @Input() size: "hero" | "compact" | "panel" = "hero";
  @Input() label = "";
  source = modeRobot("Default");
  private timer: ReturnType<typeof setInterval> | undefined;
  private media = window.matchMedia("(prefers-reduced-motion: reduce)");
  private refresh = () => this.render();
  constructor(private cdr: ChangeDetectorRef) {
    this.media.addEventListener("change", this.refresh);
    document.addEventListener("visibilitychange", this.refresh);
  }
  get description() {
    const state =
      this.phase === "success"
        ? "operation completed"
        : this.phase === "error"
          ? "needs attention"
          : this.phase === "review"
            ? "draft awaiting confirmation"
            : this.phase === "processing"
              ? "processing"
              : "ready";
    return this.intent === "idle"
      ? `${this.mode} mode robot — ${state}`
      : `${this.intent === "calendar" ? "Clock" : this.intent === "reminder" ? "Notepad" : "Thinking"} mode robot — ${state}`;
  }
  ngOnChanges() {
    this.render();
  }
  private render() {
    clearInterval(this.timer);
    const frames = this.intent === "idle" ? null : ROBOT_FRAMES[this.intent];
    if (!frames || this.phase === "idle" || this.phase === "error") {
      this.source = modeRobot(this.mode);
      this.cdr.markForCheck();
      return;
    }
    const still =
      this.phase === "success" ? 4 : this.phase === "processing" ? 2 : 3;
    if (
      this.reduced ||
      this.media.matches ||
      document.hidden ||
      this.phase === "context"
    ) {
      this.source = frames[still];
      this.cdr.markForCheck();
      return;
    }
    // Confirmation is gated by actual success; never included in a processing loop.
    const sequence =
      this.phase === "processing"
        ? [0, 1, 2, 3, 2, 1]
        : this.phase === "review"
          ? [1, 2, 3]
          : this.intent === "summarizer"
            ? [1, 2, 3, 4, 4, 4]
            : [3, 4, 4, 4];
    for (const index of new Set(sequence)) {
      const image = new Image();
      image.src = frames[index];
    }
    let position = 0;
    this.source = frames[sequence[0]];
    this.timer = setInterval(() => {
      position++;
      if (position >= sequence.length) {
        if (this.phase === "processing") position = 1;
        else {
          clearInterval(this.timer);
          return;
        }
      }
      this.source = frames[sequence[position]];
      this.cdr.markForCheck();
    }, 350);
    this.cdr.markForCheck();
  }
  fallback() {
    this.source =
      this.source === modeRobot(this.mode)
        ? "brand/mark.svg"
        : modeRobot(this.mode);
    clearInterval(this.timer);
  }
  ngOnDestroy() {
    clearInterval(this.timer);
    this.media.removeEventListener("change", this.refresh);
    document.removeEventListener("visibilitychange", this.refresh);
  }
}
