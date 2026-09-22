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
  CAPTION_HIDE_WAIT_MS,
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
      const text = await page.locator('body').innerText();
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

// The cold open starts already looking at the board, so the very first frame has
// something in it. This positioning happens before any caption is shown.
await page.locator('#lbtable').scrollIntoViewIfNeeded();
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
 */
const DRY_RUN_WAIT_MS = 60;

let failed = null;
const runStartedAt = Date.now();
try {
  for (const beat of beatsForCut(cut)) {
    const beatStartedAt = Date.now();

    for (const action of beat.actions) await runAction(page, action);
    for (const a of beat.asserts) await checkAssert(page, beat.id, a);

    const caption = captionFor(beat, cut);
    if (caption) await showCaption(page, caption);

    const dwellSec = dwellFor(beat, cut);
    const dwellMs = Math.round(dwellSec * 1000);
    if (dryRun) {
      await page.waitForTimeout(DRY_RUN_WAIT_MS);
    } else {
      const spentMs = Date.now() - beatStartedAt;
      const hideReserveMs = caption ? CAPTION_HIDE_WAIT_MS : 0;
      const remainingMs = dwellMs - spentMs - hideReserveMs;
      if (remainingMs > 0) {
        await page.waitForTimeout(remainingMs);
      } else {
        console.warn(
          `  WARN  ${beat.id} overran its dwell: dwell ${dwellSec.toFixed(1)}s, `
          + `choreography cost ${(spentMs / 1000).toFixed(2)}s `
          + `+ ${(hideReserveMs / 1000).toFixed(2)}s reserved for the caption fade-out `
          + `(over by ${(-remainingMs / 1000).toFixed(2)}s) -- waited 0s`,
        );
      }
    }
    if (caption) await hideCaption(page);

    const elapsedSec = (Date.now() - runStartedAt) / 1000;
    console.log(`  ok  ${beat.id} (${dwellSec}s, elapsed ${elapsedSec.toFixed(2)}s)`);
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
