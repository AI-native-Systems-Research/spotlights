/**
 * Replays the storyboard against the real report and records it.
 *
 * Each beat asserts its own effect before the recording moves on, so a beat that
 * silently fails to fire — a renamed id, a changed figure — fails the render
 * instead of producing a video of a broken page.
 */
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdir, rename, rm } from 'node:fs/promises';
import {
  GEOMETRY, beatsForCut, dwellFor, captionFor, validateStoryboard,
} from './storyboard.mjs';
import {
  installOverlay, showCaption, hideCaption, ring, unring, moveCursor, pulseCursor,
  beatWaits, pageText, CAPTION_MIN_ONSCREEN_MS,
} from './overlay.mjs';
import { PAGE_URL } from './paths.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, 'out');

function parseArgs(argv) {
  const cut = argv[argv.indexOf('--cut') + 1];
  if (!['gif', 'full'].includes(cut)) {
    throw new Error('usage: node scripts/demo/record.mjs --cut <gif|full> [--dry-run]');
  }
  return { cut, dryRun: argv.includes('--dry-run') };
}

async function runAction(page, action) {
  switch (action.type) {
    case 'scrollTo':
      await page.locator(action.sel).first().scrollIntoViewIfNeeded();
      await page.waitForTimeout(450);
      return;
    /**
     * `scrollTo` is scrollIntoViewIfNeeded, which does nothing at all once any sliver of
     * the element is in frame -- exactly the wrong move for something taller than the
     * viewport, which is "not needed" while 38% of it shows. This one always scrolls, to
     * the alignment the beat asks for: 'center' so an oversized element fills the frame
     * instead of hanging off the bottom of it, 'start' so a section heading sits at the
     * top of the shot with its content beneath.
     *
     * One action with a `block` option rather than a scrollCenter and a scrollTop that
     * differ by one word: the two would share every line of this body, and a third
     * alignment would then want a third near-duplicate.
     */
    case 'scrollAlign':
      await page.locator(action.sel).first()
        .evaluate((el, block) => el.scrollIntoView({ block, inline: 'nearest' }), action.block);
      await page.waitForTimeout(450);
      return;
    /**
     * The same destination as `scrollAlign`, reached in visible increments so the
     * recording contains frames at intermediate scroll positions. Two beats of one
     * expanded row read as two unrelated screens when the page cuts between them.
     *
     * The steps are driven from here, one assignment per hop, rather than handed to the
     * browser as `behavior: 'smooth'` or a CSS `scroll-behavior`: Chrome honours a
     * `prefers-reduced-motion` preference by collapsing both to an instant jump, and it
     * would do so silently -- the action would return, the assertions would pass, and the
     * video would hold the same cut this replaces. Driving it means the motion is a
     * property of the recorder, not of the recording machine's accessibility settings.
     *
     * The destination is learnt the only way that is reliable across nested scrollers:
     * ask the browser. scrollIntoView is called, the resulting scrollTops are read, and
     * the originals are put back -- all inside one `evaluate`, so the page never renders
     * the jumped state, and the frames the recorder captures start from where the
     * previous beat left it. Every scrollable ancestor is stepped, because this travel is
     * split between the leaderboard's own max-height scroller and the document.
     */
    case 'scrollStepped': {
      const steps = action.steps ?? 10;
      const stepMs = action.stepMs ?? 90;
      const plan = await page.locator(action.sel).first().evaluate((el, block) => {
        const scrollers = [];
        for (let n = el.parentElement; n; n = n.parentElement) {
          const s = getComputedStyle(n);
          const scrolls = /auto|scroll|overlay/.test(`${s.overflowY} ${s.overflowX}`);
          if (scrolls && (n.scrollHeight > n.clientHeight || n.scrollWidth > n.clientWidth)) {
            scrollers.push(n);
          }
        }
        const doc = document.scrollingElement || document.documentElement;
        if (!scrollers.includes(doc)) scrollers.push(doc);
        const from = scrollers.map((n) => n.scrollTop);
        el.scrollIntoView({ block, inline: 'nearest' });
        const to = scrollers.map((n) => n.scrollTop);
        scrollers.forEach((n, i) => { n.scrollTop = from[i]; });
        // Handed back so the steps can be set from Node, one await between each.
        window.__stepScrollers = scrollers;
        return { from, to };
      }, action.block);

      for (let i = 1; i <= steps; i += 1) {
        await page.evaluate(({ from, to, i: at, steps: n }) => {
          const scrollers = window.__stepScrollers || [];
          scrollers.forEach((el, k) => {
            el.scrollTop = from[k] + ((to[k] - from[k]) * at) / n;
          });
        }, { ...plan, i, steps });
        await page.waitForTimeout(stepMs);
      }
      await page.evaluate(() => { delete window.__stepScrollers; });
      // The last hop lands on the destination; this is the frame that holds it still.
      await page.waitForTimeout(200);
      return;
    }
    case 'click':
      await moveCursor(page, action.sel);
      await pulseCursor(page);
      await page.locator(action.sel).first().click();
      return;
    case 'select':
      await moveCursor(page, action.sel);
      await pulseCursor(page);
      await page.locator(action.sel).first().selectOption(action.value);
      return;
    case 'hover':
      await moveCursor(page, action.sel);
      await page.locator(action.sel).first().hover();
      return;
    case 'fill':
      await moveCursor(page, action.sel);
      await page.locator(action.sel).first().fill(action.text);
      return;
    case 'ring':
      await ring(page, action.sel);
      return;
    case 'unring':
      await unring(page);
      return;
    case 'pause':
      await page.waitForTimeout(action.ms);
      return;
    default:
      throw new Error(`unknown action type: ${action.type}`);
  }
}

/**
 * The two text assertions compare case-insensitively, because innerText is *rendered*
 * text: this report styles `section.sec > h2` and several headings
 * text-transform:uppercase, so "Which agent did what" reads "WHICH AGENT DID WHAT"
 * however it is spelled in the source. Patterns and expected strings are authored in
 * source case, and the storyboard is pure data that must not have to know which
 * elements the stylesheet shouts. Every figure a caption quotes is digits and
 * punctuation, which case cannot alter, so no assertion loses any strength.
 */
async function checkAssert(page, beatId, a) {
  const fail = (msg) => {
    throw new Error(`beat "${beatId}" assertion failed: ${msg}`);
  };
  switch (a.type) {
    case 'domContains': {
      // The page's own text, with the overlay excluded: the caption is up by the time
      // assertions run, so `body.innerText` would let a beat assert against its own
      // narration instead of the report. See pageText in overlay.mjs.
      const text = await pageText(page);
      if (!text.toLowerCase().includes(a.text.toLowerCase())) {
        fail(`page does not contain ${JSON.stringify(a.text)}`);
      }
      return;
    }
    case 'visible': {
      if (!(await page.locator(a.sel).first().isVisible())) fail(`${a.sel} is not visible`);
      return;
    }
    /**
     * `visible` is Playwright's isVisible(), which is true for an element parked
     * thousands of pixels above the fold: it answers "is this rendered", not "is
     * this in frame". A beat whose whole point is where the page is looking needs
     * the stronger question, so this compares the element's box against the
     * viewport and demands the box lie wholly inside it.
     */
    case 'inViewport': {
      const box = await page.locator(a.sel).first().boundingBox();
      if (!box) fail(`${a.sel} has no box, so it cannot be in the viewport`);
      const view = page.viewportSize();
      const right = box.x + box.width;
      const bottom = box.y + box.height;
      if (box.x < 0 || box.y < 0 || right > view.width || bottom > view.height) {
        fail(
          `${a.sel} is not wholly inside the ${view.width}x${view.height} viewport: `
          + `box is x ${box.x.toFixed(0)}..${right.toFixed(0)}, `
          + `y ${box.y.toFixed(0)}..${bottom.toFixed(0)}`,
        );
      }
      return;
    }
    /**
     * How much of the frame an element actually fills: the height of the intersection of
     * its box with the viewport, over the viewport's height. `inViewport` cannot answer
     * this -- it demands the whole box be inside the frame, so anything taller than the
     * viewport fails it at every scroll position, and it is silent about how much of a
     * merely-visible element is in shot. The threshold lives in the storyboard.
     */
    case 'viewportCoverage': {
      const box = await page.locator(a.sel).first().boundingBox();
      if (!box) fail(`${a.sel} has no box, so it covers none of the viewport`);
      const view = page.viewportSize();
      const bottom = box.y + box.height;
      const visibleHeight = Math.max(0, Math.min(view.height, bottom) - Math.max(0, box.y));
      const fraction = visibleHeight / view.height;
      if (fraction < a.minFraction) {
        fail(
          `${a.sel} covers ${(fraction * 100).toFixed(0)}% of the ${view.height}px viewport `
          + `height, want at least ${(a.minFraction * 100).toFixed(0)}%: `
          + `${visibleHeight.toFixed(0)}px of ${view.height}px in frame, `
          + `box is y ${box.y.toFixed(0)}..${bottom.toFixed(0)} and ${box.height.toFixed(0)}px tall`,
        );
      }
      return;
    }
    case 'attr': {
      const got = await page.locator(a.sel).first().getAttribute(a.name);
      if (got !== a.equals) fail(`${a.sel}[${a.name}] is ${JSON.stringify(got)}, want ${JSON.stringify(a.equals)}`);
      return;
    }
    case 'minChildren': {
      const n = await page.locator(a.sel).first().evaluate((el) => el.children.length);
      if (n < a.n) fail(`${a.sel} has ${n} children, want at least ${a.n}`);
      return;
    }
    case 'textMatches': {
      const texts = await page.locator(a.sel).allInnerTexts();
      if (!texts.some((t) => new RegExp(a.pattern, 'i').test(t))) {
        fail(`no ${a.sel} matches /${a.pattern}/`);
      }
      return;
    }
    default:
      fail(`unknown assert type ${a.type}`);
  }
}

const { cut, dryRun } = parseArgs(process.argv);

const problems = validateStoryboard();
if (problems.length > 0) {
  console.error('storyboard is invalid:\n  ' + problems.join('\n  '));
  process.exit(1);
}

await mkdir(OUT_DIR, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome' });
const context = await browser.newContext({
  viewport: GEOMETRY,
  deviceScaleFactor: 1,
  ...(dryRun ? {} : { recordVideo: { dir: OUT_DIR, size: GEOMETRY } }),
});
const page = await context.newPage();
await page.goto(PAGE_URL, { waitUntil: 'load' });
await installOverlay(page);

// The demo opens on the overview -- the objective in the top bar and the output tiles
// -- so the first frame is the top of the page and the opening beat's own scroll is a
// settle rather than a jump. The leaderboard is no longer pre-positioned: the beat that
// reveals it scrolls itself there. This happens before any caption is shown.
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(600);

/**
 * A beat's dwell is the *span* it is meant to occupy, not an extra pause on the end:
 * `totalDuration(cut)` sums the dwells and is the declared length of the cut. So the
 * dwell is treated as a deadline -- choreography (actions, assertions, the caption
 * fade-in), the dwell remainder *and* the closing caption fade-out all happen inside
 * it, and only the remainder is waited out. The fade-out is part of the beat, so its
 * cost is reserved up front from `CAPTION_HIDE_WAIT_MS` -- overlay.mjs owns that
 * number, and nothing here duplicates it. A beat whose choreography plus that reserve
 * is already longer than its dwell waits zero and says so, because those overruns are
 * the whole budget discussion and have to be visible to a human.
 *
 * The caption is raised BEFORE the beat's actions, and so is up for the whole beat.
 * Showing it last made its screen time whatever the dwell had left over, which for
 * every beat that overran was nothing at all: the caption appeared and the fade-out
 * immediately pulled it down. Now the choreography plays underneath the sentence that
 * describes it, and `CAPTION_MIN_ONSCREEN_MS` is the floor no caption may fall below.
 * The assertions still run after the actions -- only the caption moved -- so a beat
 * still has to prove its effect fired. Every captioned beat logs the caption's measured
 * on-screen time, because this regression is invisible in the code and shows up only in
 * a rendered GIF.
 *
 * A dry run paces nothing: it neither waits out a dwell nor honours the floor, so it
 * reports no caption time rather than a misleading one.
 */
const DRY_RUN_WAIT_MS = 60;

let failed = null;
const runStartedAt = Date.now();
try {
  for (const beat of beatsForCut(cut)) {
    const beatStartedAt = Date.now();

    // Caption first: it narrates what is about to happen, and the choreography, the
    // dwell remainder and the fade-out all run underneath it.
    const caption = captionFor(beat, cut);
    let captionShownAt = null;
    if (caption) {
      await showCaption(page, caption);
      captionShownAt = Date.now();
    }

    for (const action of beat.actions) await runAction(page, action);
    for (const a of beat.asserts) await checkAssert(page, beat.id, a);

    const dwellSec = dwellFor(beat, cut);
    const dwellMs = Math.round(dwellSec * 1000);
    if (dryRun) {
      await page.waitForTimeout(DRY_RUN_WAIT_MS);
    } else {
      const now = Date.now();
      const spentMs = now - beatStartedAt;
      const waits = beatWaits({
        dwellMs,
        spentMs,
        onScreenMs: captionShownAt === null ? 0 : now - captionShownAt,
        hasCaption: Boolean(caption),
      });
      if (waits.dwellWaitMs > 0) {
        await page.waitForTimeout(waits.dwellWaitMs);
      } else {
        console.warn(
          `  WARN  ${beat.id} overran its dwell: dwell ${dwellSec.toFixed(1)}s, `
          + `choreography cost ${(spentMs / 1000).toFixed(2)}s `
          + `+ ${(waits.hideReserveMs / 1000).toFixed(2)}s reserved for the caption fade-out `
          + `(over by ${(waits.overrunMs / 1000).toFixed(2)}s) -- waited 0s`,
        );
      }
      if (waits.floorTopUpMs > 0) {
        await page.waitForTimeout(waits.floorTopUpMs);
        console.warn(
          `  WARN  ${beat.id} held its caption ${(waits.floorTopUpMs / 1000).toFixed(2)}s `
          + `past the beat to clear the ${CAPTION_MIN_ONSCREEN_MS}ms caption floor`,
        );
      }
    }

    // Measured, not predicted: the span the caption was fully up, from the end of its
    // fade-in to the start of its fade-out.
    const onScreenMs = !dryRun && captionShownAt !== null ? Date.now() - captionShownAt : null;
    if (caption) await hideCaption(page);

    if (onScreenMs !== null && onScreenMs < CAPTION_MIN_ONSCREEN_MS) {
      console.warn(
        `  WARN  ${beat.id} caption was legible for only ${onScreenMs}ms, under the `
        + `${CAPTION_MIN_ONSCREEN_MS}ms floor -- it will flash past unread`,
      );
    }

    let captionNote = 'no caption';
    if (caption) captionNote = onScreenMs === null ? 'caption unpaced' : `caption ${onScreenMs}ms`;
    const elapsedSec = (Date.now() - runStartedAt) / 1000;
    console.log(`  ok  ${beat.id} (${dwellSec}s, ${captionNote}, elapsed ${elapsedSec.toFixed(2)}s)`);
  }
} catch (err) {
  failed = err;
}

const video = page.video();
await context.close();

if (!dryRun && video) {
  const target = resolve(OUT_DIR, `demo-${cut}.webm`);
  await rm(target, { force: true });
  await rename(await video.path(), target);
  console.log(`wrote ${target}`);
}
await browser.close();

if (failed) {
  console.error(`\n${failed.message}`);
  process.exit(1);
}
console.log(`\n${dryRun ? 'dry run' : 'recording'} of cut "${cut}" complete`);
